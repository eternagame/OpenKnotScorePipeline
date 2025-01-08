from decimal import Decimal
from timefold.solver.score import (
    HardSoftDecimalScore, ConstraintFactory, ConstraintCollectors, Joiners,
    constraint_provider
)
from .domain import Task, TaskQueue, ComputeAllocation

def define_constraints(
    soft_max_allocations: int
):
    @constraint_provider
    def _define_constraints(factory: ConstraintFactory):
        return [
            # Hard constraints
            constrain_cpus_to_present_tasks(factory),
            constrain_queue_cores_to_allocation(factory),
            constrain_queue_memory_to_allocation(factory),
            constrain_queue_gpu_to_allocation(factory),
            constrain_queue_gpu_memory_to_gpu(factory),
            constrain_queue_max_runtime_to_allocation(factory),
            # Soft constraints
            minimize_excess_allocations(factory, soft_max_allocations),
            minimize_max_cost(factory),
            minimize_avg_cost(factory),
            minimize_min_cost(factory),
        ]
    return _define_constraints

def constrain_cpus_to_present_tasks(factory: ConstraintFactory):
    # We need to include unassigned since we allow gpu_id to be unassigned
    return factory.for_each_including_unassigned(
        TaskQueue
    ).if_exists(
        # Timefold may not have assigned this a value yet - that would already be captured
        # as a negative "init" sore
        ComputeAllocation,
        Joiners.equal(lambda queue: queue.allocation, lambda allocation: allocation)
    ).filter(
        lambda queue: queue.cpus == 0 and len(queue.tasks) > 0
    ).penalize(
        HardSoftDecimalScore.ONE_HARD
    ).as_constraint(
        'A queue with tasks must have at least one CPU'
    )

def constrain_queue_cores_to_allocation(factory: ConstraintFactory):
    return factory.for_each_including_unassigned(
        TaskQueue
    ).if_exists(
        ComputeAllocation,
        Joiners.equal(lambda queue: queue.allocation, lambda allocation: allocation)
    ).group_by(
        # Sum CPUs across all queues in an allocation
        lambda queue: queue.allocation,
        ConstraintCollectors.sum(lambda queue: queue.cpus)
    ).filter(
        # Ensure the total CPUs required by queues does not exceed
        # the available CPUs
        lambda alloc, queue_cpus: queue_cpus > alloc.configuration.cpus
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda alloc, queue_cpus: queue_cpus - alloc.configuration.cpus
    ).as_constraint('Total queue CPUs <= allocation CPUs')

def constrain_queue_memory_to_allocation(factory: ConstraintFactory):
    return factory.for_each_including_unassigned(
        TaskQueue
    ).if_exists(
        ComputeAllocation,
        Joiners.equal(lambda queue: queue.allocation, lambda allocation: allocation)
    ).group_by(
        # Sum memory across all queues in an allocation
        lambda queue: queue.allocation,
        ConstraintCollectors.sum(lambda queue: queue.utilized_resources.memory)
    ).filter(
        # Ensure the total memory required by queues does not exceed
        # the available memory
        lambda alloc, queue_memory: queue_memory > alloc.configuration.memory
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        # We'll penalize in 1GB "chunks" so that memeory overages don't overwhelm the hard score
        lambda alloc, queue_memory: Decimal((queue_memory - alloc.configuration.memory) / 1024 / 1024 / 1024)
    ).as_constraint('Total queue memory <= allocation memory')

def constrain_requires_gpu(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).filter(
        lambda task: task.requires_gpu and not task.queue.accessible_resources.gpu
    ).penalize(
        HardSoftDecimalScore.ONE_HARD
    ).as_constraint(
        'GPU available for task requiring GPU'
    )

def constrain_queue_gpu_to_allocation(factory: ConstraintFactory):
    return factory.for_each_including_unassigned(
        TaskQueue
    ).if_exists(
        ComputeAllocation,
        Joiners.equal(lambda queue: queue.allocation, lambda allocation: allocation)
    ).filter(
        # If the queue is assigned a GPU, make sure it is actually an available GPU
        # (ie if we have 4 gpus, valid IDs are 0-3)
        lambda queue: queue.gpu_id is not None and queue.gpu_id <= 0 and queue.gpu_id < queue.allocation.configuration.gpus
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda queue: queue.gpu_id - queue.allocation.configuration.gpus if queue.gpu_id > 0 else -queue.gpu_id
    ).as_constraint('Queue GPU exists in allocation')

def constrain_queue_gpu_memory_to_gpu(factory: ConstraintFactory):
    return factory.for_each_including_unassigned(
        TaskQueue
    ).if_exists(
        ComputeAllocation,
        Joiners.equal(lambda queue: queue.allocation, lambda allocation: allocation)
    ).filter(
        # Only consider queues that actually use a GPU
        lambda queue: queue.gpu_id is not None
    ).group_by(
        # For each allocated GPU, sum the GPU memory
        lambda queue: (queue.allocation, queue.gpu_id),
        ConstraintCollectors.sum(lambda queue: queue.utilized_resources.gpu_memory)
    ).filter(
        # Check if the memory of all queues allocated to a GPU is more than the total GPU memory
        lambda gpu, mem: mem > gpu[0].configuration.gpu_memory
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda gpu, mem: mem - gpu[0].configuration.gpu_memory
    ).as_constraint('Total queue GPU memory <= available GPU memory')

def constrain_queue_max_runtime_to_allocation(factory: ConstraintFactory):
    return factory.for_each_including_unassigned(
        TaskQueue
    ).if_exists(
        ComputeAllocation,
        Joiners.equal(lambda queue: queue.allocation, lambda allocation: allocation)
    ).group_by(
        # Get the longest runtime across all queues in an allocation
        lambda queue: queue.allocation,
        ConstraintCollectors.max(lambda queue: queue.utilized_resources.max_runtime)
    ).filter(
        # Ensure the maximal runtime required by queues does not exceed
        # the available memory
        lambda alloc, max_runtime: max_runtime > alloc.configuration.runtime
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        # We'll penalize in 1h "chunks" so that runtime overages don't overwhelm the hard score
        lambda alloc, max_runtime: Decimal((max_runtime - alloc.configuration.runtime) / 60 / 60)
    ).as_constraint('Max runtime <= allocation runtime')

def minimize_excess_allocations(factory: ConstraintFactory, soft_max_allocations: int):
    '''
    Since we can only allocate max_allocations allocations at a time, we want to reduce
    the number of required allocations over that amount
    '''
    return factory.for_each(
        Task
    ).join(
        ComputeAllocation,
        Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
    ).group_by(
        lambda _, allocation: allocation
    ).group_by(
        ConstraintCollectors.count()
    ).filter(
        lambda count: count - soft_max_allocations > 0
    ).penalize(
        HardSoftDecimalScore.ONE_SOFT,
        lambda count: count - soft_max_allocations
    ).as_constraint('Minimize excess allocations')

def minimize_max_cost(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).join(
        ComputeAllocation,
        Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
    ).group_by(
        # Filter down to just allocations which have a task assigned to them
        lambda _, allocation: allocation
    ).penalize(
        HardSoftDecimalScore.ONE_SOFT,
        lambda allocation: allocation.configuration.compute_cost(allocation.utilized_resources.max_runtime) 
    ).as_constraint('Minimize max cost')

def minimize_avg_cost(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).join(
        ComputeAllocation,
        Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
    ).group_by(
        # Filter down to just allocations which have a task assigned to them
        lambda _, allocation: allocation
    ).penalize(
        HardSoftDecimalScore.ONE_SOFT,
        lambda allocation: allocation.configuration.compute_cost(allocation.utilized_resources.avg_runtime) 
    ).as_constraint('Minimize average cost')

def minimize_min_cost(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).join(
        ComputeAllocation,
        Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
    ).group_by(
        # Filter down to just allocations which have a task assigned to them
        lambda _, allocation: allocation
    ).penalize(
        HardSoftDecimalScore.ONE_SOFT,
        lambda allocation: allocation.configuration.compute_cost(allocation.utilized_resources.min_runtime) 
    ).as_constraint('Minimize minimum cost')
