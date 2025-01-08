from typing import Callable, Annotated, TypeVar, Type, Tuple
from dataclasses import dataclass, field
from decimal import Decimal
from timefold.solver.domain import (
    PlanningVariable, PlanningEntityCollectionProperty,
    InverseRelationShadowVariable, ShadowVariable, VariableListener,
    PlanningScore, ValueRangeProvider, PlanningId,
    planning_entity, planning_solution
)
from timefold.solver.score import HardSoftDecimalScore

# To be replaced: https://github.com/TimefoldAI/timefold-solver/pull/1271
A = TypeVar('A')
def deep_planning_clone(entity_class: Type[A] = None) -> Type[A]:
    from timefold.solver.domain._annotations import ensure_init
    ensure_init()
    from _jpyinterpreter import add_class_annotation
    from timefold.solver._timefold_java_interop import _add_to_compilation_queue
    from ai.timefold.solver.core.api.domain.solution.cloner import (DeepPlanningClone as JavaDeepPlanningClone) # type: ignore #noqa
    out = add_class_annotation(JavaDeepPlanningClone)(entity_class)
    _add_to_compilation_queue(entity_class)
    return out

class IdGenerator:
    '''
    Generates sequential IDs across the life of the program
    '''

    n = 0
    def generate(self):
        self.n += 1
        return self.n

@dataclass(eq=False)
@deep_planning_clone
class AccessibleResources:
    '''
    Describes the resources that are available when executing a task which may affect
    utilized resources
    '''

    cpus: int
    gpu: bool

@dataclass(eq=False)
@deep_planning_clone
class UtilizedResources:
    max_runtime: int
    avg_runtime: int
    min_runtime: int
    memory: int
    gpu_memory: int = 0

@dataclass
class Runnable:
    '''
    Represents a function and its arguments which we can pickle and distribute elsewhere
    '''

    func: Callable
    args: tuple
    kwargs: dict

    def run(self):
        self.func(*self.args, **self.kwargs)

    @classmethod
    def create(cls, func):
        def _create(*args, **kwargs):
            return cls(func, args, kwargs)
        return _create

@planning_entity
@dataclass(eq=False)
class Task:
    '''
    A unit of work which must be done, and the resources required to run it
    '''

    runnable: Runnable
    utilized_resources: Callable[[AccessibleResources], UtilizedResources] | UtilizedResources
    requires_gpu: bool = False

    queue: Annotated['TaskQueue', PlanningVariable] = None

    def compute_utilized_resources(self, resources: AccessibleResources) -> UtilizedResources:
        if type(self.utilized_resources) == UtilizedResources:
            return self.utilized_resources
        return self.utilized_resources(resources)

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'Task/{str(self.id)}'

class AccessibleResourcesUpdater(VariableListener):
    def after_variable_changed(self, score_director, _queue) -> None:
        queue: TaskQueue = _queue
        score_director.before_variable_changed(queue, 'accessible_resources')
        queue.accessible_resources = AccessibleResources(queue.cpus, queue.gpu_id is not None)
        score_director.after_variable_changed(queue, 'accessible_resources')

class TaskAssignmentUtillizedResourcesUpdater(VariableListener):
    # What's with this being a separate function? Something in the timefold
    # python --> java compiler is broken.
    def new_utilization_prechange(self, _task):
        task: Task = _task
        utilized = task.compute_utilized_resources(task.queue.accessible_resources)
        
        # We're no longer going to be attached to this queue, so our runtime doesn't
        # count toward the total runtime of the queue any more
        max_runtime = task.queue.utilized_resources.max_runtime - utilized.max_runtime
        avg_runtime = task.queue.utilized_resources.avg_runtime - utilized.avg_runtime
        min_runtime = task.queue.utilized_resources.min_runtime - utilized.min_runtime
        
        # If the maximum RAM/VRAM recorded for the queue is equal to the value
        # for this task, that means this task was (or was equal to) the maximum,
        # so since we're no longer a part of the queue we need to find what the next
        # lowest was
        if task.queue.utilized_resources.memory == utilized.memory:
            max_mem = 0
            for queue_task in task.queue.tasks:
                qt_utilized = queue_task.compute_utilized_resources(task.queue.accessible_resources)
                max_mem = max(max_mem, qt_utilized.memory)
        else:
            max_mem = task.queue.utilized_resources.memory
        if task.queue.utilized_resources.gpu_memory == utilized.gpu_memory:
            max_gpu_mem = 0
            for queue_task in task.queue.tasks:
                qt_utilized = queue_task.compute_utilized_resources(task.queue.accessible_resources)
                max_gpu_mem = max(max_gpu_mem, qt_utilized.gpu_memory)
        else:
            max_gpu_mem = task.queue.utilized_resources.gpu_memory
        return UtilizedResources(max_runtime, avg_runtime, min_runtime, max_mem, max_gpu_mem)

    def before_variable_changed(self, score_director, _task):
        task: Task = _task
        if task.queue != None:
            score_director.before_variable_changed(task.queue, 'utilized_resources')
            task.queue.utilized_resources = self.new_utilization_prechange(task)
            score_director.after_variable_changed(task.queue, 'utilized_resources')
    
    def new_utilization_postchange(self, _task):
        task: Task = _task
        utilized = task.compute_utilized_resources(task.queue.accessible_resources)
        
        # Extend the queue's total runtime by our task's runtime
        max_runtime = task.queue.utilized_resources.max_runtime + utilized.max_runtime
        avg_runtime = task.queue.utilized_resources.avg_runtime + utilized.avg_runtime
        min_runtime = task.queue.utilized_resources.min_runtime + utilized.min_runtime        

        # If the maximum RAM/VRAM recorded for the queue is lower than this task,
            # the max should be the utilization of this task. If it's equal or higher,
            # it's already correct
        if task.queue.utilized_resources.memory < utilized.memory:
            max_memory = utilized.memory
        else:
            max_memory = task.queue.utilized_resources.memory
        if task.queue.utilized_resources.gpu_memory < utilized.gpu_memory:
            max_gpu_memory = utilized.gpu_memory
        else:
            max_gpu_memory = task.queue.utilized_resources.gpu_memory
        
        return UtilizedResources(max_runtime, avg_runtime, min_runtime, max_memory, max_gpu_memory)

    def after_variable_changed(self, score_director, _task):
        task: Task = _task
        if task.queue != None:
            score_director.before_variable_changed(task.queue, 'utilized_resources')
            task.queue.utilized_resources = self.new_utilization_postchange(task)
            score_director.after_variable_changed(task.queue, 'utilized_resources')

class ResourceChangeUtilizedResourcesUpdater(VariableListener):
    def new_utilization(self, _queue):
        queue: TaskQueue = _queue
        utilized = UtilizedResources(0, 0, 0, 0)
        for task in queue.tasks:
            task_utilized = task.compute_utilized_resources(queue.accessible_resources)
            utilized.max_runtime += task_utilized.max_runtime
            utilized.avg_runtime += task_utilized.avg_runtime
            utilized.min_runtime += task_utilized.min_runtime
            utilized.memory = max(utilized.memory, task_utilized.memory)
            utilized.gpu_memory = max(utilized.gpu_memory, task_utilized.gpu_memory)
        return utilized

    def after_variable_changed(self, score_director, _queue):
        queue: TaskQueue = _queue
        score_director.before_variable_changed(queue, 'utilized_resources')
        queue.utilized_resources = self.new_utilization(queue)
        score_director.after_variable_changed(queue, 'utilized_resources')

@planning_entity
@dataclass(eq=False)
class TaskQueue:
    '''
    A given compute allocation may allow multiple tasks to run concurrently. A TaskQueue represents a single
    "stream" of tasks that is run serially, such that resources consumed "greedily" are constrained
    (eg, using cgroups to make a certain number of cores available) and other consumed resources never exceed
    the total capacity of the compute allocation (ie, we constrain the set of tasks such that the most
    "expensive" task in each queue of an allocation could run concurrently).
    '''
    max_cpus: int
    max_gpus: int

    cpus: Annotated[int, PlanningVariable(value_range_provider_refs=['queue_cpus'])] = 0
    gpu_id: Annotated[int | None, PlanningVariable(value_range_provider_refs=['queue_gpus'], allows_unassigned=True)] = None

    def get_cpu_range(self) -> Annotated[list[int], ValueRangeProvider(id='queue_cpus')]:
        return list(range(0, self.max_cpus + 1))
    
    def get_gpu_range(self) -> Annotated[list[int], ValueRangeProvider(id='queue_gpus')]:
        return list(range(0, self.max_gpus))

    allocation: 'ComputeAllocation' = None
    tasks: Annotated[list[Task], InverseRelationShadowVariable(source_variable_name='queue')] = field(default_factory=list)

    accessible_resources: Annotated[
        AccessibleResources,
        ShadowVariable(source_variable_name='cpus', variable_listener_class=AccessibleResourcesUpdater),
        ShadowVariable(source_variable_name='gpu_id', variable_listener_class=AccessibleResourcesUpdater),
    ] = field(default_factory=lambda: AccessibleResources(0, False))

    utilized_resources: Annotated[
        UtilizedResources,
        ShadowVariable(source_variable_name='queue', source_entity_class=Task, variable_listener_class=TaskAssignmentUtillizedResourcesUpdater),
        ShadowVariable(source_variable_name='cpus', variable_listener_class=ResourceChangeUtilizedResourcesUpdater),
        ShadowVariable(source_variable_name='gpu_id', variable_listener_class=ResourceChangeUtilizedResourcesUpdater),
    ] = field(default_factory=lambda: UtilizedResources(0, 0, 0, 0))

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'TaskQueue/{str(self.id)}(cpus: {self.cpus}, gpu_id: {self.gpu_id})'
    
class AllocationUtilizedResourcesUpdater(VariableListener):
    def before_variable_changed(self, score_director, _queue) -> None:
        queue: TaskQueue = _queue

        new_utilized = UtilizedResources(
            queue.allocation.utilized_resources.max_runtime,
            queue.allocation.utilized_resources.avg_runtime,
            queue.allocation.utilized_resources.min_runtime,
            queue.allocation.utilized_resources.memory,
            queue.allocation.utilized_resources.gpu_memory,
        )
        new_utilized.memory -= queue.utilized_resources.memory
        new_utilized.gpu_memory -= queue.utilized_resources.gpu_memory
        if queue.allocation.utilized_resources.max_runtime == queue.utilized_resources.max_runtime:
            new_utilized.max_runtime = max(
                alloc_queue.utilized_resources.max_runtime for alloc_queue in queue.allocation.queues
            )
        if queue.allocation.utilized_resources.avg_runtime == queue.utilized_resources.avg_runtime:
            new_utilized.avg_runtime = max(
                alloc_queue.utilized_resources.avg_runtime for alloc_queue in queue.allocation.queues
            )
        if queue.allocation.utilized_resources.min_runtime == queue.utilized_resources.min_runtime:
            new_utilized.min_runtime = max(
                alloc_queue.utilized_resources.min_runtime for alloc_queue in queue.allocation.queues
            )

        score_director.before_variable_changed(queue.allocation, 'utilized_resources')
        queue.allocation.utilized_resources = new_utilized
        score_director.after_variable_changed(queue.allocation, 'utilized_resources')
    
    def after_variable_changed(self, score_director, _queue) -> None:
        queue: TaskQueue = _queue

        new_utilized = UtilizedResources(
            queue.allocation.utilized_resources.max_runtime,
            queue.allocation.utilized_resources.avg_runtime,
            queue.allocation.utilized_resources.min_runtime,
            queue.allocation.utilized_resources.memory,
            queue.allocation.utilized_resources.gpu_memory,
        )
        new_utilized.memory += queue.utilized_resources.memory
        # It's a little silly to be adding GPU memory across all GPUs, but for the sake
        # of being able to reuse the same class, we'll do this. We're not actively
        # using it anyways, and at least it gives us some info about total GPU
        # utilization if we want it
        new_utilized.gpu_memory += queue.utilized_resources.gpu_memory
        # The maximum runtime for all queues is the longest maximum - worst case, we have to wait
        # for the longest worst case
        if queue.allocation.utilized_resources.max_runtime < queue.utilized_resources.max_runtime:
            new_utilized.max_runtime = queue.utilized_resources.max_runtime
        # This one is a little weird. What's the expected runtime of the combination of
        # multiple independent streams? Consider queue averages of 0, 10, and 20. That means
        # typically the first queue will finish, then the second, then the third - so the
        # third (the longest) is the runtime we expect everythig to take
        if queue.allocation.utilized_resources.avg_runtime < queue.utilized_resources.avg_runtime:
            new_utilized.avg_runtime = queue.utilized_resources.avg_runtime
        # The minimum runtime for all queues is the longest minimum - worst case, we have to wait
        # for the longest best case
        if queue.allocation.utilized_resources.min_runtime < queue.utilized_resources.min_runtime:
            new_utilized.min_runtime = queue.utilized_resources.min_runtime

        score_director.before_variable_changed(queue.allocation, 'utilized_resources')
        queue.allocation.utilized_resources = new_utilized
        score_director.after_variable_changed(queue.allocation, 'utilized_resources')

@planning_entity
@dataclass
class ComputeAllocation:
    '''
    An available set of resources which tasks can be assigned to and run on. This could be a Slurm job,
    VPS, or so forth. 
    '''

    queues: list[TaskQueue] = field(default_factory=list)
    configuration: Annotated['ComputeConfiguration', PlanningVariable] = None

    utilized_resources: Annotated[
        UtilizedResources,
        ShadowVariable(source_variable_name='utilized_resources', source_entity_class=TaskQueue, variable_listener_class=AllocationUtilizedResourcesUpdater),
    ] = field(default_factory=lambda: UtilizedResources(0, 0, 0, 0))

    def nonempty_queues(self):
        return list(
            queue for queue in self.queues if len(queue.tasks) > 0
        )

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'ComputeAllocation/{str(self.id)}(configuration: {self.configuration.id})'

@planning_entity
@dataclass(eq=False)
class ComputeConfiguration:
    '''
    A description of resources available to a ComputeAllocation where the specific amount of resources
    can be optimized. To represent something like a possible VPS instance type, you can specify
    resource properties as a single value. To represent something like a Slurm job where you can
    more granularly allocate resources, you can specify a range or list of options
    '''

    cpu_range: int | list[int] | range
    memory_range: int | list[int] | range
    gpu_range: int | list[int] | range
    runtime_range: int | list[int] | range
    # We assume allocated GPUs are homogenious, which is true in all cases I can think of right now
    gpu_memory: int
    cost: Callable[['ComputeConfiguration', int], int | Decimal] | int | Decimal
    '''
    Function which takes the current compute configuration and a given runtime
    and returns the cost
    '''

    def compute_cost(self, runtime: int):
        if type(self.cost) in [int, Decimal]:
            return self.cost
        return self.cost(self, runtime)
    
    def cpu_value_range(self) -> Annotated[list[int], ValueRangeProvider(id='config_cpus')]:
        if type(self.cpu_range) == range:
            return list(self.cpu_range)
        elif type(self.cpu_range) == int:
            return [self.cpu_range]
        else:
            return self.cpu_range
    
    def memory_value_range(self) -> Annotated[list[int], ValueRangeProvider(id='config_memory')]:
        if type(self.memory_range) == range:
            if self.memory_range.step == 1:
                # Because memory would have so many potential values, we'll cap it to 1024 "steps" unless otherwise
                # specified so we don't blow out memory unnecessarily.
                return list(range(self.memory_range.start, self.memory_range.stop, (self.memory_range.stop - self.memory_range.start) // 1024))
            return list(self.memory_range)
        elif type(self.memory_range) == int:
            return [self.memory_range]
        else:
            return self.memory_range

    def gpu_value_range(self) -> Annotated[list[int], ValueRangeProvider(id='config_gpus')]:
        if type(self.gpu_range) == range:
            return list(self.gpu_range)
        elif type(self.gpu_range) == int:
            return [self.gpu_range]
        else:
            return self.gpu_range
        
    def runtime_value_range(self) -> Annotated[list[int], ValueRangeProvider(id='config_runtime')]:
        if type(self.runtime_range) == range:
            if self.runtime_range.step == 1:
                # Because runtime would have so many potential values, we'll cap it to 1024 "steps" unless otherwise
                # specified so we don't blow out memory unnecessarily.
                # For a timeout of 7 days, that's ~10m increments
                return list(range(self.runtime_range.start, self.runtime_range.stop, (self.runtime_range.stop - self.runtime_range.start) // 1024))
        elif type(self.runtime_range) == int:
            return [self.runtime_range]
        else:
            return self.runtime_range
    
    cpus: Annotated[int, PlanningVariable(value_range_provider_refs=['config_cpus'])] = 1
    memory: Annotated[int, PlanningVariable(value_range_provider_refs=['config_memory'])] = 0
    gpus: Annotated[int, PlanningVariable(value_range_provider_refs=['config_gpus'])] = 0
    runtime: Annotated[int, PlanningVariable(value_range_provider_refs=['config_runtime'])] = 0

    allocations: Annotated[list[ComputeAllocation], InverseRelationShadowVariable(source_variable_name='configuration')] = field(default_factory=list)

    def nonempty_allocations(self):
        return list(
            alloc for alloc in self.allocations if any(
                True for queue in alloc.queues if len(queue.tasks) > 0
            )
        )

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'ComputeConfiguration/{self.id}(cpus: {self.cpus}, memory: {self.memory}, gpus: {self.gpus}, runtime: {self.runtime})'

@planning_solution
@dataclass(eq=False)
class Schedule:
    compute_configurations: Annotated[list[ComputeConfiguration], PlanningEntityCollectionProperty, ValueRangeProvider]
    compute_allocations: Annotated[list[ComputeAllocation], PlanningEntityCollectionProperty, ValueRangeProvider]
    task_queues: Annotated[list[TaskQueue], PlanningEntityCollectionProperty, ValueRangeProvider]
    tasks: Annotated[list[Task], PlanningEntityCollectionProperty]

    def nonempty_compute_configurations(self):
        return list(
            config for config in self.compute_configurations
            if any(True for alloc in config.allocations if any(
                True for queue in alloc.queues if len(queue.tasks) > 0
            ))
        )
    
    def nonempty_compute_allocations(self):
        return list(
            alloc for alloc in self.compute_allocations if any(
                True for queue in alloc.queues if len(queue.tasks) > 0
            )
        )

    score: Annotated[HardSoftDecimalScore | None, PlanningScore] = None
