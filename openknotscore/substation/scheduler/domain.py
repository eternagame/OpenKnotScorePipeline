from typing import Callable, Generic, TypeVar
from dataclasses import dataclass, field

class IdGenerator:
    '''
    Generates sequential IDs across the life of the program
    '''

    n = 0
    def generate(self):
        self.n += 1
        return self.n


GPUMemoryType = TypeVar('GPUMemoryType', int, list[int])
@dataclass
class UtilizedResources(Generic[GPUMemoryType]):
    max_runtime: int
    avg_runtime: int
    min_runtime: int
    cpus: int
    memory: int
    gpu_memory: GPUMemoryType

    def freeze(self):
        return FrozenUtilizedResources[GPUMemoryType](
            self.max_runtime,
            self.avg_runtime,
            self.min_runtime,
            self.cpus,
            self.memory,
            self.gpu_memory
        )

@dataclass(frozen=True)
class FrozenUtilizedResources(Generic[GPUMemoryType]):
    max_runtime: int
    avg_runtime: int
    min_runtime: int
    cpus: int
    memory: int
    gpu_memory: GPUMemoryType

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

@dataclass
class Task:
    '''
    A unit of work which must be done, and the resources required to run it
    '''

    runnable: Runnable
    utilized_resources: UtilizedResources[int]

    queue: 'TaskQueue' = None

    id: int = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'Task/{str(self.id)}(resources={self.utilized_resources})'

@dataclass
class TaskQueue:
    '''
    A given compute allocation may allow multiple tasks to run concurrently. A TaskQueue represents a single
    "stream" of tasks that is run serially, such that resources consumed "greedily" are constrained
    (eg, using cgroups to make a certain number of cores available) and other consumed resources never exceed
    the total capacity of the compute allocation (ie, we constrain the set of tasks such that the most
    "expensive" task in each queue of an allocation could run concurrently).
    '''
    gpu_id: int | None

    allocation: 'ComputeAllocation'
    tasks: list[Task] = field(default_factory=list)
    child_queues: list['TaskQueue'] = field(default_factory=list)
    parent_queue: 'TaskQueue' = None
    utilized_resources: UtilizedResources[int] = field(default_factory=lambda: UtilizedResources[int](0,0,0,0,0,0))
    chain_utilized_resources: UtilizedResources[int] = field(default_factory=lambda: UtilizedResources[int](0,0,0,0,0,0))
    '''
    Resources utilized starting from the root parent through child queues leading to and including this queue
    '''

    def leaf_queues(self) -> list['TaskQueue']:
        if len(self.child_queues) == 0:
            return [self]

        leaves: list['TaskQueue'] = []
        for queue in self.child_queues:
            if len(queue.child_queues) == 0:
                leaves.append(queue)
            else:
                leaves.extend(queue.leaf_queues())
        return leaves

    def tasks_recursive(self):
        for task in self.tasks:
            yield task
        
        for child in self.child_queues:
            for task in child.tasks_recursive():
                yield task

    id: int = field(default_factory=IdGenerator().generate)

    def __repr__(self):
        return f'TaskQueue/{str(self.id)}(tasks: {len(self.tasks)}, gpu_id: {self.gpu_id})'

@dataclass
class ComputeAllocation:
    '''
    An available set of resources which tasks can be assigned to and run on. This could be a Slurm job,
    VPS, or so forth. 
    '''
    configuration: 'ComputeConfiguration'
    queues: list[TaskQueue] = field(default_factory=list)
    utilized_resources: UtilizedResources[list[int]] = field(default_factory=lambda: UtilizedResources[list[int]](0,0,0,0,0,[]))

    def nonempty_queues(self):
        return list(
            queue for queue in self.queues if len(queue.tasks) > 0
        )

    def leaf_queues(self) -> list['TaskQueue']:
        leaves: list['TaskQueue'] = []
        for queue in self.queues:
            leaves.extend(queue.leaf_queues())
        return leaves

    id: int = field(default_factory=IdGenerator().generate)

    def tasks(self):
        for queue in self.queues:
            for task in queue.tasks_recursive():
                yield task

    def __post_init__(self):
        self.utilized_resources.gpu_memory = [0 for _ in range(self.configuration.gpus)]

    def __repr__(self):
        return f'ComputeAllocation/{str(self.id)}(configuration: {self.configuration.id})'

@dataclass
class ComputeConfiguration:
    '''
    A description of resources available to a ComputeAllocation, eg as defined by a VPS instance
    type or Slurm job
    '''

    cpus: int
    memory: int
    gpus: int
    runtime: int
    # We assume allocated GPUs are homogenious, which is true in all cases I can think of right now
    gpu_memory: int

    allocations: list[ComputeAllocation] = field(default_factory=list)
    max_utilized_resources: UtilizedResources[list[int]] = field(default_factory=lambda: UtilizedResources[list[int]](0, 0, 0, 0, 0, []))

    def nonempty_allocations(self):
        return list(
            alloc for alloc in self.allocations if any(
                True for queue in alloc.queues if len(queue.tasks) > 0
            )
        )

    id: int = field(default_factory=IdGenerator().generate)

    def __post_init__(self):
        self.max_utilized_resources.gpu_memory = [0 for _ in range(self.gpus)]

    def __repr__(self):
        return f'ComputeConfiguration/{self.id}(cpus: {self.cpus}, memory: {self.memory}, gpus: {self.gpus}, runtime: {self.runtime})'

@dataclass
class Schedule:
    compute_configurations: list[ComputeConfiguration]
    compute_allocations: list[ComputeAllocation]
    task_queues: list[TaskQueue]
    tasks: list[Task]

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
