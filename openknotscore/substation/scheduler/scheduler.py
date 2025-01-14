from dataclasses import replace
from functools import cmp_to_key
import math
from .domain import Task, TaskQueue, ComputeAllocation, ComputeConfiguration, Schedule, UtilizedResources

def prioritize(a: UtilizedResources, b: UtilizedResources):
    # We order which tasks we schedule first based on how we want to
    # prioritize unused resources and how hard it is to schedule them.
    
    # We start by scheduling GPU tasks before tasks that use no GPUs. GPUs are
    # expensive, so we want to prioritize use of available GPU instances for
    # tasks that actually require GPUs, and then "fill in" with CPU-only tasks
    # if they fit.
    # We then turn to CPUs, as generally CPUs will be more scarce and more expensive than memory.
    # Runtime is last, as we can always just run an allocation for less time without
    # wasting any resources. We focus on max runtime before average runtime because
    # max runtime is a hard constraint, whereas average runtime just allows for
    # further optimization

    # In all cases, use of larger resources are scheduled before use of smaller
    # resources since they're "harder" to schedule - eg, on a 12-core, 1h allocation
    # you can fit 5 consecutive 20m 2-core tasks alongside a 1h 10-core task,
    # but if the 2-core tasks were scheduled concurrently before the 10-core task,
    # the larger task couldn't be fit (and even if it did, there would be more wasted space)

    if a.gpu_memory > b.gpu_memory:
        return -1
    if a.gpu_memory < b.gpu_memory:
        return 1
    
    if a.cpus > b.cpus:
        return -1
    if a.cpus < b.cpus:
        return 1
    
    if a.memory > b.memory:
        return -1
    if a.memory < b.memory:
        return 1
    
    if a.max_runtime > b.max_runtime:
        return -1
    if a.max_runtime < b.max_runtime:
        return 1
    
    if a.avg_runtime > b.avg_runtime:
        return -1
    if a.avg_runtime < b.avg_runtime:
        return 1
    
    return 0

def get_new_allocation(resource_request: UtilizedResources, schedule: Schedule):
    for config in schedule.compute_configurations:
        if (
            config.cpus >= resource_request.cpus
            and config.gpu_memory >= resource_request.gpu_memory
            and config.memory > resource_request.memory
            and config.runtime > resource_request.max_runtime
        ):
            alloc = ComputeAllocation(config)
            schedule.compute_allocations.append(alloc)
            config.allocations.append(alloc)
            return alloc
    raise RuntimeError(f'Could not schedule task - required resources ({resource_request}) could not fit into any configuration')

def get_gpu_slot(allocation: ComputeAllocation):
    # We choose the GPU with the lowest utilization because that gives us the best likihood
    # that we'll be able to maximize the utilization of all GPUs with future (smaller) tasks.
    best_id, best_mem = None, math.inf
    for id, mem in enumerate(allocation.utilized_resources.gpu_memory):
        if mem < best_mem:
            best_id = id
            best_mem = mem
    return best_id

def recover_resources(resource_request: UtilizedResources, queue: TaskQueue):
    # Note we don't do any runtime checks in this function. If we're recovering available
    # resources from a parent queue, that queue obviously ended before us. If we're recovering
    # from the allocation, those resources have never been allocated to anything
    available_resources = replace(queue.utilized_resources)
    candidate_queue = queue.parent_queue
    while candidate_queue != None:
        # When we split a queue into child queues, if there are resources left over
        # which we haven't assigned a task to yet, they will be allocated to an empty queue
        for maybe_unused_queue in candidate_queue.child_queues:
            if len(maybe_unused_queue.tasks) == 0:
                available_resources.cpus += maybe_unused_queue.utilized_resources.cpus
                available_resources.memory += maybe_unused_queue.utilized_resources.memory
                available_resources.gpu_memory += maybe_unused_queue.utilized_resources.gpu_memory
        candidate_queue = candidate_queue.parent_queue
    available_resources.cpus += queue.allocation.configuration.cpus - queue.allocation.utilized_resources.cpus
    available_resources.memory += queue.allocation.configuration.memory - queue.allocation.utilized_resources.memory
    if queue.gpu_id:
        available_resources.gpu_memory += queue.allocation.configuration.gpu_memory - queue.allocation.utilized_resources.gpu_memory[queue.gpu_id]

    if (
        available_resources.cpus < resource_request.cpus
        or available_resources.memory < resource_request.memory
        or available_resources.gpu_memory < resource_request.gpu_memory
    ):
        return False
    
    candidate_queue = queue.parent_queue
    while candidate_queue != None:
        for maybe_unused_queue in candidate_queue.child_queues:
            if len(maybe_unused_queue.tasks) == 0:
                if queue.utilized_resources.cpus < resource_request.cpus:
                    cpus_to_recover = min(
                        resource_request.cpus - queue.utilized_resources.cpus,
                        maybe_unused_queue.utilized_resources.cpus
                    )
                    queue.utilized_resources.cpus += cpus_to_recover
                    maybe_unused_queue.utilized_resources.cpus -= cpus_to_recover
                if queue.utilized_resources.memory < resource_request.memory:
                    memory_to_recover = min(
                        resource_request.memory - queue.utilized_resources.memory,
                        maybe_unused_queue.utilized_resources.memory
                    )
                    queue.utilized_resources.memory += memory_to_recover
                    maybe_unused_queue.utilized_resources.memory -= memory_to_recover
                if queue.utilized_resources.gpu_memory < resource_request.gpu_memory:
                    gpu_memory_to_recover = min(
                        resource_request.gpu_memory - queue.utilized_resources.gpu_memory,
                        maybe_unused_queue.utilized_resources.gpu_memory
                    )
                    queue.utilized_resources.gpu_memory += gpu_memory_to_recover
                    maybe_unused_queue.utilized_resources.gpu_memory -= gpu_memory_to_recover
    if queue.utilized_resources.cpus < resource_request.cpus:
        cpus_to_recover = resource_request.cpus - queue.utilized_resources.cpus
        queue.utilized_resources.cpus += cpus_to_recover
        queue.allocation.utilized_resources.cpus += cpus_to_recover
    if queue.utilized_resources.memory < resource_request.memory:
        memory_to_recover = resource_request.memory - queue.utilized_resources.memory
        queue.utilized_resources.memory += memory_to_recover
        queue.allocation.utilized_resources.memory += memory_to_recover
    if queue.utilized_resources.gpu_memory < resource_request.gpu_memory:
        gpu_memory_to_recover = resource_request.gpu_memory - queue.utilized_resources.gpu_memory
        queue.utilized_resources.gpu_memory += gpu_memory_to_recover
        queue.allocation.utilized_resources.gpu_memory[queue.gpu_id] += gpu_memory_to_recover
    
    return True

def slots_for_task(resource_request: UtilizedResources, allocation: ComputeAllocation, schedule: Schedule):
    # If there is room for a new queue to fit this task, create it. We want to prefer
    # filling up more of the allocation resources rather than extending the runtime,
    # as we can relinquish an allocation early but not shrink its resources.
    #
    # NB: This can lead to different inefficiencies. Eg, imagine you have two identical 6-core
    # tasks that are placed concurrently. Then you have two 5-core tasks and a 2-core task.
    # Because we assign resources to queues based on the maximum the queue consumes
    # (as our runtime estimates for each task are only estimates, we cant be sure when
    # the resources will become available - otherwise we'd have to wait on tasks in other
    # queues creating a DAG, but that adds more complexity and has its own drawbacks
    # of inserting wasted space plus some later task may take longer-than-average
    # but we would otherwise have gotten a head start on said task),
    # we can't place these three new tasks side by side - at best we can place the
    # two 5-core tasks side by side and insert a one-core task next to each of them
    # if we can. This, however, seems preferable to the scenario where we only use, say,
    # 20% of the allocation's resources for its entire available timeout
    while (
        allocation.configuration.cpus - allocation.utilized_resources.cpus >= resource_request.cpus
        and allocation.configuration.gpu_memory - min(allocation.utilized_resources.gpu_memory + [0]) >= resource_request.gpu_memory
        and allocation.configuration.memory - allocation.utilized_resources.memory >= resource_request.memory
    ):
        queue = TaskQueue(
            get_gpu_slot(allocation) if resource_request.gpu_memory > 0 else None,
            allocation
        )
        queue.utilized_resources.cpus = resource_request.cpus
        queue.utilized_resources.gpu_memory = resource_request.gpu_memory
        queue.utilized_resources.memory = resource_request.memory
        allocation.queues.append(queue)
        allocation.utilized_resources.cpus += resource_request.cpus
        if queue.gpu_id: allocation.utilized_resources.gpu_memory[queue.gpu_id] += resource_request.gpu_memory
        allocation.utilized_resources.memory += resource_request.memory
        schedule.task_queues.append(queue)
        yield queue
    
    # We prefer to shorten the average runtime of all queue chains, though the hard constraint is
    # that the chain's maximum runtime must be <= the allocation's runtime limit
    leaf_queues = sorted(
        [queue for queue in allocation.leaf_queues()],
        key=lambda queue: queue.chain_utilized_resources.avg_runtime
    )
    while leaf_queues:
        if (
            len(leaf_queues) > 1
            and leaf_queues[0].utilized_resources.avg_runtime > leaf_queues[1].utilized_resources.avg_runtime
        ):
            # We inserted a task that made this queue no longer the smallest. Re-sort!
            leaf_queues.sort(key=lambda queue: queue.chain_utilized_resources.avg_runtime)
        
        queue = leaf_queues[0]
        if (
            queue.utilized_resources.cpus < resource_request.cpus
            or queue.utilized_resources.memory < resource_request.memory
            or queue.utilized_resources.gpu_memory < resource_request.gpu_memory
            or queue.chain_utilized_resources.max_runtime + resource_request.max_runtime > allocation.configuration.runtime
        ):
            # First see if there are unallocated resoruces we can use to make the queue fit this task
            resize_success = recover_resources(resource_request, queue)
            if not resize_success:
                # This queue cant fit any more of this task - remove it from consideration
                leaf_queues.pop(0)
                continue

        # Note this is if not elif - we may have just eg recovered some CPUs to fit the task,
        # but already had more than enough memory
        # Also note that we still shrink the queue even if we'd be left over with eg juts
        # memory but no CPUs, which we wouldn't actually be able to allocate a task on.
        # This doesn't prevent future jobs down the line from using those resources, since
        # our resource recovery algorithm will pick those up if it can
        if (
            queue.chain_utilized_resources.cpus > resource_request.cpus
            or queue.chain_utilized_resources.memory > resource_request.memory
            or queue.chain_utilized_resources.gpu_memory > resource_request.gpu_memory
        ):
            # We can potentially run multiple concurrent tasks in this queue, since it was
            # initially sized for a larger task. Split the queue into smaller parts. To reduce the
            # amount we need to keep track of various subtrees, we'll always "fill" the remainder
            # of the resources in queue with another queue even though we don't have a task for
            # it yet, instead if a task wants to use it and its too large but there's no tasks
            # in it, we can just resize it.
            available_resources = queue.utilized_resources
            if len(queue.tasks) == 0:
                queue.utilized_resources = UtilizedResources(
                    0, 0, 0,
                    resource_request.cpus,
                    resource_request.memory, 
                    resource_request.gpu_memory
                )
                new_queue = TaskQueue(queue.gpu_id, queue.allocation)
                new_queue.chain_utilized_resources = replace(queue.chain_utilized_resources)
                new_queue.utilized_resources.cpus = available_resources.cpus - queue.utilized_resources.cpus
                new_queue.utilized_resources.memory = available_resources.memory - queue.utilized_resources.memory
                new_queue.utilized_resources.gpu_memory = available_resources.gpu_memory - queue.utilized_resources.gpu_memory
                queue.parent_queue.child_queues.append(new_queue)
                schedule.task_queues.append(new_queue)
                leaf_queues.insert(1, new_queue)
            else:
                queue_a = TaskQueue(queue.gpu_id, allocation)
                queue_a.chain_utilized_resources = replace(queue.chain_utilized_resources)
                queue_a.utilized_resources.cpus = resource_request.cpus
                queue_a.utilized_resources.memory = resource_request.memory
                queue_a.utilized_resources.gpu_memory = resource_request.gpu_memory

                queue_b = TaskQueue(queue.gpu_id, allocation)
                queue_b.chain_utilized_resources = replace(queue.chain_utilized_resources)
                queue_b.utilized_resources.cpus = available_resources.cpus - queue_a.utilized_resources.cpus
                queue_b.utilized_resources.memory = available_resources.memory - queue_a.utilized_resources.memory
                queue_b.utilized_resources.gpu_memory = available_resources.gpu_memory - queue_a.utilized_resources.gpu_memory

                queue.child_queues = [queue_a, queue_b]
                schedule.task_queues.extend([queue_a, queue_b])
                leaf_queues.insert(0, queue_a)
                leaf_queues.insert(1, queue_b)
                queue = queue_a

        yield queue

def schedule_task(task: Task, queue: TaskQueue):
    task.queue = queue
    queue.tasks.append(task)
    queue.utilized_resources.min_runtime +=  task.utilized_resources.min_runtime
    queue.utilized_resources.avg_runtime +=  task.utilized_resources.avg_runtime
    queue.utilized_resources.max_runtime +=  task.utilized_resources.max_runtime
    queue.chain_utilized_resources.min_runtime +=  task.utilized_resources.min_runtime
    queue.chain_utilized_resources.avg_runtime +=  task.utilized_resources.avg_runtime
    queue.chain_utilized_resources.max_runtime +=  task.utilized_resources.max_runtime
    queue.allocation.utilized_resources.min_runtime +=  task.utilized_resources.min_runtime
    queue.allocation.utilized_resources.avg_runtime +=  task.utilized_resources.avg_runtime
    queue.allocation.utilized_resources.max_runtime +=  task.utilized_resources.max_runtime

def fill_allocation(
    allocation: ComputeAllocation,
    task_buckets: dict[UtilizedResources, list[Task]],
    scheduling_priorities: list[UtilizedResources],
    schedule: Schedule
):
    for bucket in scheduling_priorities:
        slots = slots_for_task(bucket, allocation, schedule)
        tasks = task_buckets[bucket]
        while (queue := next(slots, None)) and tasks:
            schedule_task(tasks.pop(), queue)

def schedule_tasks(
    tasks: list[Task],
    compute_configurations: list[ComputeConfiguration]
) -> Schedule:
    schedule = Schedule(
        [replace(config) for config in compute_configurations],
        [],
        [],
        [replace(task) for task in tasks],
    )

    task_buckets: dict[UtilizedResources, list[Task]] = {}
    for task in schedule.tasks:
        task_buckets.setdefault(task.utilized_resources.freeze(), []).append(task)

    scheduling_priorities: list[UtilizedResources] = sorted(task_buckets.keys(), key=cmp_to_key(prioritize))
    for bucket in scheduling_priorities:
        while task_buckets[bucket]:
            allocation = get_new_allocation(bucket, schedule)
            fill_allocation(allocation, task_buckets, scheduling_priorities, schedule)

    return schedule
