from typing import Callable, Annotated, Generic, TypeVar, Type, TYPE_CHECKING
from dataclasses import dataclass, field
import math
from decimal import Decimal
from timefold.solver.domain import (
    PlanningVariable, PlanningEntityCollectionProperty, ProblemFactCollectionProperty,
    InverseRelationShadowVariable, ShadowVariable, PiggybackShadowVariable, VariableListener,
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

TaskDetails = TypeVar('TaskDetails')

class IdGenerator:
    n = 0
    def generate(self):
        self.n += 1
        return self.n

@planning_entity
@dataclass
class Task(Generic[TaskDetails]):
    '''
    An individual unit of work that needs to be performed and its resource requirements
    '''
    
    details: TaskDetails

    max_utilized_runtime: int | Callable[['ResourceConfiguration'], int]
    '''
    NOTE: Due to limitations in timefold, this must be a reference to a fuction NOT a lambda
    if it is a callable
    '''
    max_utilized_mem_mb: int | Callable[['ResourceConfiguration'], int]
    '''
    NOTE: Due to limitations in timefold, this must be a reference to a fuction NOT a lambda
    if it is a callable
    '''
    max_utilized_gpu_mem_mb: int

    shard: Annotated['ComputeShard', PlanningVariable] = None

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def calc_max_utilized_mem_mb(self, resources: 'ResourceConfiguration'):
        if type(self.max_utilized_mem_mb) == int:
            return self.max_utilized_mem_mb
        else:
            return self.max_utilized_mem_mb(resources)

    def calc_max_utilized_runtime(self, resources: 'ResourceConfiguration'):
        if type(self.max_utilized_runtime) == int:
            return self.max_utilized_runtime
        else:
            return self.max_utilized_runtime(resources)

    def __repr__(self):
        return f'Task/{str(self.id)}'

@planning_entity
@dataclass
class ResourceConfiguration:
    '''
    An amount of available resources to a task, as configured in a ComputeShard
    '''

    cpus: Annotated[int, PlanningVariable(value_range_provider_refs = ['cpu_range'])] = 0
    gpu: Annotated[bool, PlanningVariable(value_range_provider_refs = ['gpu_range'])] = False

    def get_cpu_range(self) -> Annotated[list[int], ValueRangeProvider(id='cpu_range')]:
        return list(range(1, self.shard.allocation.max_cores+1))

    def get_gpu_range(self) -> Annotated[list[bool], ValueRangeProvider(id='gpu_range')]:
        return [True, False] if self.shard.allocation.allow_gpu else [False]

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)
    
    if TYPE_CHECKING:
        # We can't properly declare this field because of https://github.com/TimefoldAI/timefold-solver/issues/1282
        shard: 'ComputeShard' = None

    def __repr__(self):
        return f'ResourceConfiguration/{str(self.id)}(cpus: {self.cpus}, gpu: {self.gpu})'

class TaskAssignmentRuntimeListener(VariableListener):
    def before_variable_changed(self, score_director, _task) -> None:
        # Type annotations in the function definition causes the Java interop to break for some reason,
        # presumably due to https://github.com/TimefoldAI/timefold-solver/issues/1282, so we'll type it here instead
        task: Task = _task
        if task.shard != None:
            score_director.before_variable_changed(task.shard, 'max_runtime')
            task.shard.max_runtime -= task.calc_max_utilized_runtime(task.shard.resources)
            score_director.after_variable_changed(task.shard, 'max_runtime')

    def after_variable_changed(self, score_director, _task) -> None:
        task: Task = _task
        if task.shard != None:
            score_director.before_variable_changed(task.shard, 'max_runtime')
            task.shard.max_runtime += task.calc_max_utilized_runtime(task.shard.resources)
            score_director.after_variable_changed(task.shard, 'max_runtime')

class ResourceConfigurationRuntimeListener(VariableListener):
    def after_variable_changed(self, score_director, _resources):
        resources: ResourceConfiguration = _resources
        score_director.before_variable_changed(resources.shard, 'max_runtime')
        resources.shard.max_runtime = sum(
            task.calc_max_utilized_runtime(resources) for task in resources.shard.tasks
        )
        score_director.after_variable_changed(resources.shard, 'max_runtime')

class TaskAssignmentMemoryListener(VariableListener):
    def before_variable_changed(self, score_director, _task) -> None:
        task: Task = _task
        if task.shard != None:
            if task.calc_max_utilized_mem_mb(task.shard.resources) == task.shard.max_memory:
                score_director.before_variable_changed(task.shard, 'max_memory')
                task.shard.max_memory = max(
                    (
                        shard_task.calc_max_utilized_mem_mb(task.shard.resources)
                        for shard_task in task.shard.tasks if shard_task != task
                    ),
                    default = 0
                )
                score_director.after_variable_changed(task.shard, 'max_memory')

    def after_variable_changed(self, score_director, _task) -> None:
        task: Task = _task
        if task.shard != None:
            task_mem = task.calc_max_utilized_mem_mb(task.shard.resources)
            if task_mem > task.shard.max_memory:
                score_director.before_variable_changed(task.shard, 'max_memory')
                task.shard.max_memory = task_mem
                score_director.after_variable_changed(task.shard, 'max_memory')

class ResourceConfigurationMemoryListener(VariableListener):
    def after_variable_changed(self, score_director, _resources):
        resources: ResourceConfiguration = _resources
        score_director.before_variable_changed(resources.shard, 'max_memory')
        resources.shard.max_memory = max(
            (
                shard_task.calc_max_utilized_mem_mb(resources)
                for shard_task in resources.shard.tasks
            ),
            default = 0
        )
        score_director.after_variable_changed(resources.shard, 'max_memory')

class TaskAssignmentGpuMemoryListener(VariableListener):
    def before_variable_changed(self, score_director, _task) -> None:
        task: Task = _task
        if task.shard != None and task.shard.resources.gpu:
            if task.max_utilized_gpu_mem_mb == task.shard.max_memory:
                score_director.before_variable_changed(task.shard, 'max_gpu_memory')
                task.shard.max_gpu_memory = max(
                    (
                        shard_task.max_utilized_gpu_mem_mb
                        for shard_task in task.shard.tasks if shard_task != task
                    ),
                    default = 0
                )
                score_director.after_variable_changed(task.shard, 'max_gpu_memory')

    def after_variable_changed(self, score_director, _task) -> None:
        task: Task = _task
        if task.shard != None and task.shard.resources.gpu:
            task_gpu_mem = task.max_utilized_gpu_mem_mb
            if task_gpu_mem > task.shard.max_gpu_memory:
                score_director.before_variable_changed(task.shard, 'max_gpu_memory')
                task.shard.max_gpu_memory = task_gpu_mem
                score_director.after_variable_changed(task.shard, 'max_gpu_memory')

class ResourceConfigurationGpuMemoryListener(VariableListener):
    def after_variable_changed(self, score_director, _resources):
        resources: ResourceConfiguration = _resources
        score_director.before_variable_changed(resources.shard, 'max_gpu_memory')
        if resources.gpu:
            resources.shard.max_gpu_memory = max(
                (
                    shard_task.max_utilized_gpu_mem_mb
                    for shard_task in resources.shard.tasks
                ),
                default = 0
            )
        else:
            resources.shard.max_gpu_memory = 0
        score_director.after_variable_changed(resources.shard, 'max_gpu_memory')

@planning_entity
@dataclass
class ComputeShard:
    '''
    A slice of resources from a ComputeAllocation, and the schedule of tasks to be run serially on it

    We introduce this mechanism of "sharding" as opposed to just scheduling on the ComputeAllocation
    directly so that we can ensure each task has the resources it needs while optimizing how well
    "packed" the schedule is (minimizing wasted resources), while keeping things simpler than
    if we constructed a DAG of tasks (which additionally while potentially more flexible and
    able to pack more granularly, would not allow for effective re-use of resources when tasks
    end early, as the resources would need to be reserved for the next scheduled task or else potentially
    harm the efficiency of the scheduling if some other task that fits were scheduled but locks
    up resources in a poor way)
    '''
    resources: 'ResourceConfiguration'
    allocation: 'ComputeAllocation'
    tasks: Annotated[list[Task], InverseRelationShadowVariable(source_variable_name='shard')] = field(default_factory=list)

    max_runtime: Annotated[
        int,
        ShadowVariable(source_variable_name='shard', source_entity_class=Task, variable_listener_class=TaskAssignmentRuntimeListener),
        ShadowVariable(source_variable_name='cpus', source_entity_class=ResourceConfiguration, variable_listener_class=ResourceConfigurationRuntimeListener),
        ShadowVariable(source_variable_name='gpu', source_entity_class=ResourceConfiguration, variable_listener_class=ResourceConfigurationRuntimeListener)
    ] = 0
    max_memory: Annotated[
        int,
        ShadowVariable(source_variable_name='shard', source_entity_class=Task, variable_listener_class=TaskAssignmentMemoryListener),
        ShadowVariable(source_variable_name='cpus', source_entity_class=ResourceConfiguration, variable_listener_class=ResourceConfigurationMemoryListener),
        ShadowVariable(source_variable_name='gpu', source_entity_class=ResourceConfiguration, variable_listener_class=ResourceConfigurationMemoryListener)
    ] = 0
    max_gpu_memory: Annotated[
        int,
        ShadowVariable(source_variable_name='shard', source_entity_class=Task, variable_listener_class=TaskAssignmentGpuMemoryListener),
        ShadowVariable(source_variable_name='gpu', source_entity_class=ResourceConfiguration, variable_listener_class=ResourceConfigurationGpuMemoryListener)
    ] = 0

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'ComputeShard/{self.id}({self.resources})'

class ComputeShardRuntimeListener(VariableListener):
    def after_variable_changed(self, score_director, _shard):
        shard: ComputeShard = _shard
        if shard.max_runtime > shard.allocation.requested_timeout:
            score_director.before_variable_changed(shard.allocation, 'requested_timeout')
            shard.allocation.requested_timeout = shard.max_runtime
            score_director.after_variable_changed(shard.allocation, 'requested_timeout')
        elif shard.max_runtime < shard.allocation.requested_timeout:
            score_director.before_variable_changed(shard.allocation, 'requested_timeout')
            shard.allocation.requested_timeout = max(
                allocshard.max_runtime for allocshard in shard.allocation.shards
            )
            score_director.after_variable_changed(shard.allocation, 'requested_timeout')

class ResourceConfigurationCpuListener(VariableListener):
    def before_variable_changed(self, score_director, _resources):
        resources: ResourceConfiguration = _resources
        score_director.before_variable_changed(resources.shard.allocation, 'requested_cores')
        resources.shard.allocation.requested_cores -= resources.cpus
        score_director.after_variable_changed(resources.shard.allocation, 'requested_cores')
    
    def after_variable_changed(self, score_director, _resources):
        resources: ResourceConfiguration = _resources
        score_director.before_variable_changed(resources.shard.allocation, 'requested_cores')
        resources.shard.allocation.requested_cores += resources.cpus
        score_director.after_variable_changed(resources.shard.allocation, 'requested_cores')

class ComputeShardMemoryListener(VariableListener):
    def before_variable_changed(self, score_director, _shard):
        shard: ComputeShard = _shard
        score_director.before_variable_changed(shard.allocation, 'requested_memory')
        shard.allocation.requested_memory -= shard.max_memory
        score_director.after_variable_changed(shard.allocation, 'requested_memory')
    
    def after_variable_changed(self, score_director, _shard):
        shard: ComputeShard = _shard
        score_director.before_variable_changed(shard.allocation, 'requested_memory')
        shard.allocation.requested_memory += shard.max_memory
        score_director.after_variable_changed(shard.allocation, 'requested_memory')

class AllocationResourceConfigurationGpuListener(VariableListener):
    def after_variable_changed(self, score_director, _resources):
        resources: ResourceConfiguration = _resources
        if resources.gpu and resources.shard.allocation.requested_gpus < 1:
            score_director.before_variable_changed(resources.shard.allocation, 'requested_gpus')
            resources.shard.allocation.requested_gpus = 1
            score_director.after_variable_changed(resources.shard.allocation, 'requested_gpus')
        elif not resources.gpu and resources.shard.allocation.requested_gpus > 0:
            score_director.before_variable_changed(resources.shard.allocation, 'requested_gpus')
            resources.shard.allocation.requested_gpus = 1 if any(
                shard.resources.gpu for shard in resources.shard.allocation.shards
            ) else 0
            score_director.after_variable_changed(resources.shard.allocation, 'requested_gpus')

@deep_planning_clone
@dataclass
class ComputeAllocation:
    '''
    An available set of compute resources which tasks can be run on
    All resources should be collocated on a single machine (ie, such that it could all
    be available to one process)
    '''
    max_cores: int
    cpu_cost: int | Decimal

    allow_gpu: bool
    max_gpu_mem_mb: int
    gpu_cost: int | Decimal

    max_mem_per_core_mb: int
    mem_mb_cost: int | Decimal

    max_timeout_secs: int

    shards: list[ComputeShard] = field(default_factory=list)

    id: Annotated[int, PlanningId] = field(default_factory=IdGenerator().generate)

    requested_timeout: Annotated[
        int,
        ShadowVariable(source_variable_name='max_runtime', source_entity_class=ComputeShard, variable_listener_class=ComputeShardRuntimeListener)
    ] = 0
    requested_cores: Annotated[
        int,
        ShadowVariable(source_variable_name='cpus', source_entity_class=ResourceConfiguration, variable_listener_class=ResourceConfigurationCpuListener)
    ] = 0
    requested_memory: Annotated[
        int,
        ShadowVariable(source_variable_name='max_memory', source_entity_class=ComputeShard, variable_listener_class=ComputeShardMemoryListener)
    ] = 0
    requested_gpus: Annotated[
        int,
        ShadowVariable(source_variable_name='gpu', source_entity_class=ResourceConfiguration, variable_listener_class=AllocationResourceConfigurationGpuListener)
    ] = 0

    def allocated_cores(self):
        return max(self.requested_cores, math.ceil(self.requested_memory / self.max_mem_per_core_mb))

    def max_allocable_mem_mb(self):
        return self.max_mem_per_core_mb * self.max_cores
    
    def __repr__(self):
        return f'ComputeAllocation/{str(self.id)}'

@planning_solution
@dataclass
class Schedule:
    tasks: Annotated[list[Task], PlanningEntityCollectionProperty, ValueRangeProvider]
    compute_allocations: Annotated[list[ComputeAllocation], ProblemFactCollectionProperty]
    compute_shards: Annotated[list[ComputeShard], PlanningEntityCollectionProperty, ValueRangeProvider]
    resource_configurations: Annotated[list[ResourceConfiguration], PlanningEntityCollectionProperty]

    score: Annotated[HardSoftDecimalScore | None, PlanningScore] = None
