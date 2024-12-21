import math
from timefold.solver.score import (
    HardSoftDecimalScore, ConstraintFactory, ConstraintCollectors, Joiners, BiConstraintStream,
    constraint_provider
)
from .domain import Task, ComputeShard, ComputeAllocation

# TODO: Support a given task requiring a GPU
def define_constraints(
    soft_max_allocations: int
):
    @constraint_provider
    def _define_constraints(factory: ConstraintFactory):
        return [
            # Hard constraints
            constrain_allocation_max_cores(factory),
            constrain_allocation_max_gpus(factory),
            constrain_allocation_max_gpu_mem(factory),
            constrain_allocation_max_mem(factory),
            constrain_allocation_max_timeout(factory),
            # Soft constraints
            minimize_excess_allocations(factory, soft_max_allocations),
            minimize_max_cost(factory),
        ]
    return _define_constraints

def constrain_allocation_max_cores(factory: ConstraintFactory):
    return factory.for_each(
        ComputeAllocation
    ).filter(
        lambda allocation: allocation.allocated_cores() > allocation.max_cores
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda allocation: allocation.allocated_cores() - allocation.max_cores
    ).as_constraint('Maximum cores per allocation')

def constrain_allocation_max_gpus(factory: ConstraintFactory):
    return factory.for_each(
        ComputeAllocation
    ).filter(
        lambda allocation: allocation.requested_gpus > 0 and not allocation.allow_gpu
    ).penalize(
        HardSoftDecimalScore.ONE_HARD
    ).as_constraint('Maximum gpus per allocation')

def constrain_allocation_max_gpu_mem(factory: ConstraintFactory):
    '''
    Ensure we'd never use more GPU memory than we have access to
    '''
    return factory.for_each(
        ComputeShard
    ).group_by(
        lambda shard:  shard.allocation,
        ConstraintCollectors.sum(lambda shard: shard.max_gpu_memory)
    ).filter(
        lambda allocation, total_max_gpu_mem: total_max_gpu_mem > allocation.max_gpu_mem_mb
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda allocation, total_max_gpu_mem: total_max_gpu_mem - allocation.max_gpu_mem_mb
    ).as_constraint('Maximum GPU memory per allocation')

def constrain_allocation_max_mem(factory: ConstraintFactory):
    '''
    Ensure we'd never use more memory than we have access to
    '''
    return factory.for_each(
        ComputeAllocation
    ).filter(
        lambda allocation: allocation.requested_memory > allocation.max_allocable_mem_mb()
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda allocation: allocation.requested_memory - allocation.max_allocable_mem_mb()
    ).as_constraint('Maximum memory per allocation')

def constrain_allocation_max_timeout(factory: ConstraintFactory):
    '''
    Ensure no shard attempts to run longer than the maximum timeout
    '''
    return factory.for_each(
        ComputeAllocation
    ).filter(
        lambda allocation: allocation.requested_timeout > allocation.max_timeout_secs
    ).penalize(
        HardSoftDecimalScore.ONE_HARD,
        lambda allocation: allocation.requested_timeout - allocation.max_timeout_secs
    ).as_constraint('Maximum timeout per allocation')

def minimize_excess_allocations(factory: ConstraintFactory, soft_max_allocations: int):
    '''
    Since we can only allocate max_allocations allocations at a time, we want to reduce
    the number of required allocations over that amount
    '''
    return factory.for_each(
        Task
    ).join(
        ComputeAllocation,
        Joiners.equal(lambda task: task.shard.allocation, lambda allocation: allocation)
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
        ComputeAllocation
    ).penalize(
        HardSoftDecimalScore.ONE_SOFT,
        lambda allocation: (
            allocation.allocated_cores() * allocation.cpu_cost
            + allocation.requested_gpus * allocation.gpu_cost
            + allocation.requested_memory * allocation.mem_mb_cost
        ) * allocation.requested_timeout
    ).as_constraint('Minimize cost')
