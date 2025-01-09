from abc import ABC, abstractmethod
import math
from timefold.solver import SolverFactory
from timefold.solver.config import SolverConfig, ScoreDirectorFactoryConfig, TerminationConfig
from .domain import Schedule, Task, TaskQueue, ComputeAllocation, ComputeConfiguration
from .constraints import define_constraints

class MaxAllocations(ABC):
    @abstractmethod
    def calculate(self, tasks: list[Task], max_timeout_secs: int):
        pass

class FixedMaxAllocations(MaxAllocations):
    def __init__(self, allocations: int):
        super().__init__()
        self.allocations = allocations

    def calculate(self, tasks: list[Task], max_timeout_secs: int):
        return self.allocations

class ExhaustiveMaxAllocations(MaxAllocations):
    def calculate(self, tasks: list[Task], max_timeout_secs: int):
        return len(tasks)
    
class ConservativeMaxAllocations(MaxAllocations):
    '''
    Determines the number of max allocations by calculating the number of bins required
    if we used a next-fit packing algorithm
    '''

    def __init__(self, relaxation_ratio: int | float = 1):
        super().__init__()
        self.relaxation_ratio = relaxation_ratio

    def calculate(self, tasks: list[Task], max_timeout_secs: int) -> int:
        return math.ceil(
            sum([
                task.utilized_resources.max_runtime for task in tasks
            ]) / max_timeout_secs * self.relaxation_ratio
        )

class LinearBlendMaxAllocations(MaxAllocations):
    '''
    Performs a linear interpolation between ConservativeMaxAllocations and ExhaustiveMaxAllocations
    '''

    def __init__(self, blend_ratio: float):
        super().__init__()
        self.blend_ratio = blend_ratio

    def calculate(self, tasks: list[Task], max_timeout_secs: int):
        a = ConservativeMaxAllocations().calculate(tasks, max_timeout_secs)
        b = ExhaustiveMaxAllocations().calculate(tasks, max_timeout_secs)
        return math.ceil(
            a + ((b - a) * self.blend_ratio)
        )

class AcceleratingBlendMaxAllocations(MaxAllocations):
    '''
    Interpolates between ConservativeMaxAllocations and ExhaustiveMaxAllocations such that
    it starts at a specific interpolation ratio but the ratio increases inversely with the distance
    between the two - ie, the closer ConservativeMaxAllocations and ExhaustiveMaxAllocations
    are, the closer the result will be to ExhaustiveMaxAllocations
    '''

    def __init__(self, min_blend_ratio: float, blend_ratio_increase_factor: int | float):
        super().__init__()
        self.min_blend_ratio = min_blend_ratio
        self.blend_ratio_increase_factor = blend_ratio_increase_factor

    def calculate(self, tasks: list[Task], max_timeout_secs: int):
        a = ConservativeMaxAllocations().calculate(tasks, max_timeout_secs)
        b = ExhaustiveMaxAllocations().calculate(tasks, max_timeout_secs)

        # A proportion of how close our low value is to our high value
        progress = a/b
        # We raise to the power of 1/increase_factor as opposed to just increase_factor because progress
        # will already be a fraction and so an exponent > 1 would make it go down, but we want a
        # larger blend ratio to make the number bigger not smaller
        blend_ratio_increase = progress ** (1 / self.blend_ratio_increase_factor)
        # Factor in the starting blend ratio, and then for the remaining amount we could potentially
        # increase the blend ratio increase it by the proportion we just determined (ie, when a=0
        # the blend_ratio is min_blend_ratio, but by the time a=b it should be 1, so we take that
        # 0.9 and multiply it by our increase function - we're doing a second interpolation here
        # where blend_ratio increase is itself a blend ratio!)
        blend_ratio = self.min_blend_ratio + (1 - self.min_blend_ratio) * blend_ratio_increase

        return  math.ceil(
            a + (b - a) * blend_ratio
        )
    
def solve(
    tasks: list[Task],
    compute_configurations: list[ComputeConfiguration],
    soft_max_allocations: int,
    hard_max_allocations: MaxAllocations = AcceleratingBlendMaxAllocations(0.005, 2)
) -> Schedule:
    solver_factory = SolverFactory.create(
        SolverConfig(
            solution_class=Schedule,
            entity_class_list=[Task, TaskQueue, ComputeAllocation, ComputeConfiguration],
            score_director_factory_config=ScoreDirectorFactoryConfig(
                constraint_provider_function=define_constraints(soft_max_allocations)
            ),
            termination_config=TerminationConfig(
                best_score_feasible=True
            )
        )
    )
    solver = solver_factory.build_solver()

    max_timeout = max(max(conf.runtime_value_range()) for conf in compute_configurations)
    max_cpus = max(max(conf.cpu_value_range()) for conf in compute_configurations)
    max_gpus = max(max(conf.gpu_value_range()) if len(conf.gpu_value_range()) > 0 else 0 for conf in compute_configurations)

    # timefold doesn't give us a way to dynamically allocate entities, so we need to pre-instantiate
    # a fixed number of compute allocations to fit jobs into. The absolute worst case would be
    # one allocation per task. However, if we have, say, a million tasks (especially small ones), 
    # that would be rather excessive and likely to make it harder for timefold to converge on a solution
    # in a reasonable amount of time. We can do better.
    #
    # If we used next-fit 1D bin packing, that would give us an approximation ratio of 2 (ie, we'd need up to 
    # 2x the number of bins vs the number of bins we would need if we could instead slice
    # items to perfectly fit into bins). We may want to use more bins than that if, say, there are many different
    # task sizes in terms of eg memory use and we want to "under-fill" allocations and instead make sure resource
    # requirements closely fit the allocation to reduce cost.
    #
    # We should upper-bound on the number of tasks since we won't use any more than that anyways.
    # Beyond that it seems reasonable to lower-bound on our soft max since we're ok with using that
    # many anyways. Beyond that, we'll let the user choose a value if they want, but I've come up with
    # a default heuristic where the numbers seem "reasonable" by eye
    #
    # Note that we don't further reduce the expected number of allocations based on eg the number of cores
    # available per allocation. This is for simplicity's sake, particularly given how varried the resource
    # requirements of any given job may be
    #
    # A lot of this is gut feeling
    concrete_max_allocations = min(
        max(
            hard_max_allocations.calculate(tasks, max_timeout),
            soft_max_allocations,
        ),
        len(tasks)
    )

    compute_allocations = []
    task_queues = []
    for _ in range(concrete_max_allocations):
        alloc = ComputeAllocation()
        for _ in range(max_cpus):
            queue = TaskQueue(max_gpus=max_gpus)
            queue.allocation = alloc
            alloc.queues.append(queue)
            task_queues.append(queue)
        compute_allocations.append(alloc)
    
    problem = Schedule(
        compute_configurations,
        compute_allocations,
        task_queues,
        tasks
    )
    return solver.solve(problem)
