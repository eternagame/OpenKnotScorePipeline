from typing import Callable
from decimal import Decimal
from timefold.solver.score import (
    HardSoftDecimalScore, ConstraintFactory, ConstraintCollectors, Joiners,
    constraint_provider
)
from .domain import Task, TaskQueue, ComputeAllocation, ComputeConfiguration

def define_constraints(
    soft_max_allocations: int
):
    @constraint_provider
    def _define_constraints(factory: ConstraintFactory):
        return [
            # Hard constraints
            constrain_queue_cores_to_allocation(factory),
            constrain_queue_memory_to_allocation(factory),
            constrain_requires_gpu(factory),
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

def constrain_queue_cores_to_allocation(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).group_by(
        lambda task: task.queue,
        ConstraintCollectors.max(lambda task: task.utilized_resources.cpus)
    ).group_by(
        lambda queue, cpus: queue.allocation,
        ConstraintCollectors.sum(lambda queue, cpus: cpus)
    ).filter(
        lambda alloc, cpus: alloc.configuration != None and cpus > alloc.configuration.cpus
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda alloc, cpus: cpus - alloc.configuration.cpus
    ).as_constraint('Total queue CPUs <= allocation CPUs')

def constrain_queue_memory_to_allocation(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).group_by(
        lambda task: task.queue,
        ConstraintCollectors.max(lambda task: task.utilized_resources.memory)
    ).group_by(
        lambda queue, memory: queue.allocation,
        ConstraintCollectors.sum(lambda queue, memory: memory)
    ).filter(
        lambda alloc, memory: alloc.configuration != None and memory > alloc.configuration.memory
    ).penalize_decimal(
        HardSoftDecimalScore.ONE_HARD,
        lambda alloc, memory: Decimal((memory - alloc.configuration.memory) / 1024 / 1024 / 1024)
    ).as_constraint('Total queue memory <= allocation memory')

def constrain_requires_gpu(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).filter(
        lambda task: task.utilized_resources.gpu_memory > 0 and task.queue.gpu_id is None
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
    return factory.for_each(
        Task
    ).group_by(
        lambda task: task.queue,
        ConstraintCollectors.max(lambda task: task.utilized_resources.gpu_memory)
    ).group_by(
        lambda queue, memory: queue.allocation,
        lambda queue, memory: queue.gpu_id,
        ConstraintCollectors.sum(lambda queue, memory: memory)
    ).filter(
        lambda alloc, gpu_id, memory: alloc.configuration != None and memory > alloc.configuration.gpu_memory
    ).penalize_decimal(
        HardSoftDecimalScore.ONE_HARD,
        lambda alloc, gpu_id, memory: Decimal((memory - alloc.configuration.gpu_memory) / 1024 / 1024 / 1024)
    ).as_constraint('Total queue GPU memory <= available GPU memory')

def constrain_queue_max_runtime_to_allocation(factory: ConstraintFactory):
    return factory.for_each(
        Task
    ).group_by(
        lambda task: task.queue,
        ConstraintCollectors.sum(lambda task: task.utilized_resources.max_runtime)
    ).group_by(
        lambda queue, runtime: queue.allocation,
        ConstraintCollectors.max(lambda queue, runtime: runtime)
    ).filter(
        lambda alloc, runtime: alloc.configuration != None and runtime > alloc.configuration.runtime
    ).penalize_decimal(
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

def minimize_cost(factory: ConstraintFactory, get_runtime: Callable[[Task], int]):
    return factory.for_each(
        Task
    ).if_exists(
        ComputeAllocation,
        Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
    ).filter(
        lambda task: task.queue.allocation.configuration.fine_grained_resources == True
    ).group_by(
        lambda task: task.queue,
        ConstraintCollectors.max(lambda task: task.utilized_resources.cpus)
    ).group_by(
        lambda queue, cpus: queue.allocation,
        ConstraintCollectors.sum(lambda queue, cpus: cpus),
    ).group_by(
        lambda alloc, cpus: alloc.configuration,
        ConstraintCollectors.count_bi(),
        ConstraintCollectors.max(lambda allolc, cpus: cpus),
    ).map(
        lambda config, alloc_count, cpus: config,
        lambda config, alloc_count, cpus: (alloc_count, cpus,)
    ).join(
        factory.for_each_including_unassigned(
            TaskQueue
        ).if_exists(
            ComputeAllocation,
            Joiners.equal(lambda queue: queue.allocation, lambda allocation: allocation)
        ).filter(
            lambda queue: queue.allocation.configuration.fine_grained_resources == True
        ).group_by(
            lambda queue: (queue.allocation, queue.gpu_id),
        ).group_by(
            lambda allocation_gpu: allocation_gpu[0],
            ConstraintCollectors.count()
        ).group_by(
            lambda alloc, gpus: alloc.configuration,
            ConstraintCollectors.max(lambda alloc, gpus: gpus)
        ).map(lambda config, gpus: (config, gpus - 1)),
        Joiners.equal(lambda config, resources: config, lambda config_gpus: config_gpus[0])
    ).map(
        lambda config, resources, config_gpus: config,
        lambda config, resources, config_gpus: (*resources, config_gpus[1])
    ).join(
        factory.for_each(
            Task
        ).if_exists(
            ComputeAllocation,
            Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
        ).filter(
            lambda task: task.queue.allocation.configuration.fine_grained_resources == True
        ).group_by(
            lambda task: task.queue,
            ConstraintCollectors.max(lambda task: task.utilized_resources.memory)
        ).group_by(
            lambda queue, memory: queue.allocation,
            ConstraintCollectors.sum(lambda queue, memory: memory)
        ).group_by(
            lambda alloc, memory: alloc.configuration,
            ConstraintCollectors.max(lambda alloc, memory: memory)
        ).map(lambda config, mem: (config, mem)),
        Joiners.equal(lambda config, resources: config, lambda config_mem: config_mem[0])
    ).map(
        lambda config, resources, config_mem: config,
        lambda config, resources, config_mem: (*resources, config_mem[1])
    ).join(
        factory.for_each(
            Task
        ).if_exists(
            ComputeAllocation,
            Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
        ).filter(
            lambda task: task.queue.allocation.configuration.fine_grained_resources == True
        ).group_by(
            lambda task: task.queue,
            ConstraintCollectors.sum(get_runtime)
        ).group_by(
            lambda queue, runtime: queue.allocation,
            ConstraintCollectors.max(lambda queue, runtime: runtime)
        ).group_by(
            lambda alloc, cpus: alloc.configuration,
            ConstraintCollectors.max(lambda alloc, runtime: runtime)
        ).map(lambda config, runtime: (config, runtime)),
        Joiners.equal(lambda config, resources: config, lambda config_runtime: config_runtime[0])
    ).map(
        lambda config, resources, config_runtime: config,
        lambda config, resources, config_runtime: (*resources, config_runtime[1])
    ).concat(
        factory.for_each(
            Task
        ).if_exists(
            ComputeAllocation,
            Joiners.equal(lambda task: task.queue.allocation, lambda allocation: allocation)
        ).filter(
            lambda task: task.queue.allocation.configuration.fine_grained_resources == False
        ).group_by(
            lambda task: task.queue,
            ConstraintCollectors.sum(lambda task: task.utilized_resources.avg_runtime)
        ).group_by(
            lambda queue, runtime: queue.allocation,
            ConstraintCollectors.max(lambda queue, runtime: runtime)
        ).group_by(
            lambda alloc, runtime: alloc.configuration,
            ConstraintCollectors.count_bi(),
            ConstraintCollectors.max(lambda alloc, runtime: runtime)
        ).map(
            lambda config, alloc_count, runtime: config,
            lambda config, alloc_count, runtime: (alloc_count, config.cpus, config.gpus, config.memory, runtime)
        )
    ).penalize_decimal(
        HardSoftDecimalScore.ONE_SOFT,
        lambda config, resources: config.compute_cost(*resources[1:]) * resources[0]
    )

def minimize_max_cost(factory: ConstraintFactory):
    return minimize_cost(
        factory, lambda task: task.utilized_resources.max_runtime
    ).as_constraint('Minimize max cost')

def minimize_avg_cost(factory: ConstraintFactory):
    return minimize_cost(
        factory, lambda task: task.utilized_resources.avg_runtime
    ).as_constraint('Minimize average cost')

def minimize_min_cost(factory: ConstraintFactory):
    return minimize_cost(
        factory, lambda task: task.utilized_resources.min_runtime
    ).as_constraint('Minimize min cost')
