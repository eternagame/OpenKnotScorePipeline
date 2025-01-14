from typing import TYPE_CHECKING, NamedTuple
from os import path
from abc import ABC, abstractmethod
from ..db import DB
from ..scheduler.domain import Task, TaskQueue, Schedule

class DBComputeConfiguration(NamedTuple):
    allocations: list[int]

class DBAllocation(NamedTuple):
    queues: list[int]

class DBQueue(NamedTuple):
    tasks: list[int]
    cpus: int
    gpu_id: int | None
    child_queues: list[int]

class DBTask(NamedTuple):
    function: int
    args: list[int]
    kwargs: list[int]

class Runner(ABC):
    @abstractmethod
    def run(self, tasks: list[Task], job_name: str):
        pass

    @abstractmethod
    def forecast(self, tasks: list[Task], config):
        pass

    @staticmethod
    def serialize_tasks(schedule: Schedule, job_name: str, db_path: str):
        dbpath = path.join(db_path, f'taskinfo-{job_name}.sbstdb')
        with DB(dbpath) as db:
            def add_queue(queue: TaskQueue):
                task_ids = []
                for task in queue.tasks:
                    func_id = db.insert('functions', task.runnable.func)
                    arg_ids = [db.insert('args', arg) for arg in task.runnable.args]
                    kwarg_ids = [db.insert('kwargs', kwarg) for kwarg in task.runnable.kwargs.items()]
                    task_ids.append(db.insert('tasks', DBTask(func_id, arg_ids, kwarg_ids)))
                qid = db.insert('queues', DBQueue(
                    task_ids,
                    queue.utilized_resources.cpus,
                    queue.gpu_id,
                    [add_queue(child_queue) for child_queue in queue.child_queues]
                ))
                return qid

            for compute_config in schedule.nonempty_compute_configurations():
                alloc_ids = []
                for alloc in compute_config.nonempty_allocations():
                    queue_ids = []
                    for queue in alloc.nonempty_queues():
                        queue_ids.append(add_queue(queue))
                    alloc_ids.append(db.insert('allocations', DBAllocation(queue_ids)))
                db.insert('compute_configurations', DBComputeConfiguration(alloc_ids), compute_config.id)
        
        return dbpath

    @staticmethod
    def run_serialized_queue(dbpath: str, id: bytes):
        with DB(dbpath) as db:
            queue: DBQueue = db.get('queues', id)
            for task_id in queue.tasks:
                task: DBTask = db.get('tasks', task_id)
                func = db.get('functions', task.function)
                args = [db.get('args', arg_id) for arg_id in task.args]
                kwargs = {k:v for (k,v) in (db.get('kwargs', kwarg_id) for kwarg_id in task.kwargs)}
                func(*args, **kwargs)
