import unittest
from .scheduler import schedule_tasks
from .domain import Task, ComputeConfiguration, Runnable, UtilizedResources

class TestSchedule(unittest.TestCase):
    def test_gpu_nonexclusive(self):
        donothing = Runnable.create(lambda: 1)()
        tasks = [
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
        ]
        configs = [
            ComputeConfiguration(cpus=4, memory=400, runtime=1000, gpus=1, gpu_memory=1000)
        ]
        schedule = schedule_tasks(tasks, configs, False)
        self.assertEqual(len(schedule.compute_allocations), 1)
        self.assertEqual(len(schedule.compute_allocations[0].queues), 4)
        self.assertEqual(len(schedule.compute_allocations[0].queues[0].tasks), 2)
        self.assertEqual(len(schedule.compute_allocations[0].queues[1].tasks), 1)
        self.assertEqual(len(schedule.compute_allocations[0].queues[2].tasks), 1)
        self.assertEqual(len(schedule.compute_allocations[0].queues[3].tasks), 1)
    
    def test_gpu_exclusive(self):
        donothing = Runnable.create(lambda: 1)()
        tasks = [
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
            Task(donothing, UtilizedResources(10, 10, 10, cpus=1, memory=100, gpu_memory=100)),
        ]
        configs = [
            ComputeConfiguration(cpus=4, memory=400, runtime=1000, gpus=1, gpu_memory=1000)
        ]
        schedule = schedule_tasks(tasks, configs, True)
        self.assertEqual(len(schedule.compute_allocations), 1)
        self.assertEqual(len(schedule.compute_allocations[0].queues), 1)
        self.assertEqual(len(schedule.compute_allocations[0].queues[0].tasks), 5)

if __name__ == '__main__':
    unittest.main()
