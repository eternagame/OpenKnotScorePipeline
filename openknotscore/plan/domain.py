from typing import Callable, Annotated, TypeVar, Type
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
class UtilizedResources:
    max_runtime: int
    avg_runtime: int
    min_runtime: int
    cpus: int
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
    utilized_resources: UtilizedResources

    queue: Annotated['TaskQueue', PlanningVariable] = None

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'Task/{str(self.id)}'

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
    max_gpus: int

    allocation: 'ComputeAllocation' = None
    tasks: Annotated[list[Task], InverseRelationShadowVariable(source_variable_name='queue')] = field(default_factory=list)

    gpu_id: Annotated[int | None, PlanningVariable(value_range_provider_refs=['queue_gpus'], allows_unassigned=True)] = None

    def get_gpu_range(self) -> Annotated[list[int], ValueRangeProvider(id='queue_gpus')]:
        return list(range(0, self.max_gpus))

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'TaskQueue/{str(self.id)}(tasks: {len(self.tasks)}, gpu_id: {self.gpu_id})'
    
class AllocationUtilizedResourcesUpdater(VariableListener):
    def before_variable_changed(self, score_director, _queue) -> None:
        queue: TaskQueue = _queue

        new_utilized = UtilizedResources(
            queue.allocation.utilized_resources.max_runtime,
            queue.allocation.utilized_resources.avg_runtime,
            queue.allocation.utilized_resources.min_runtime,
            queue.allocation.utilized_resources.cpus,
            queue.allocation.utilized_resources.memory,
            queue.allocation.utilized_resources.gpu_memory,
        )
        new_utilized.cpus -= queue.utilized_resources.cpus
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
            queue.allocation.utilized_resources.cpus,
            queue.allocation.utilized_resources.memory,
            queue.allocation.utilized_resources.gpu_memory,
        )
        new_utilized.cpus += queue.utilized_resources.cpus
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

    cpus: int
    memory: int
    gpus: int
    runtime: int
    # We assume allocated GPUs are homogenious, which is true in all cases I can think of right now
    gpu_memory: int
    cost: Callable[
        [
            Annotated[int, 'cpus'],
            Annotated[int, 'gpus'],
            Annotated[int, 'memory'],
            Annotated[int, 'runtime']
        ],
        int | Decimal
    ] | int | Decimal
    '''
    Function which takes the current compute configuration and a given runtime
    and returns the cost
    '''
    fine_grained_resources: bool
    '''
    Whether this configuration's resources can be requested to a specific amount.
    If False, the cost will be computed assuming exactly config.cpus, etc
    will be allocated for exactly config.runtime. If true, we will instead compute
    the cost assuming the number of cores, etc allocated and the runtime will be the
    maximum required across all allocations, simply treating config.cpus, etc the max
    amount that can be requested 
    '''

    def compute_cost(self, cpus: int, gpus: int, memory: int, runtime: int):
        if type(self.cost) in [int, Decimal]:
            return self.cost
        return self.cost(cpus, gpus, memory, runtime)

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
