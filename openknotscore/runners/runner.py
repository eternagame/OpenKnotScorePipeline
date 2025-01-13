from typing import TYPE_CHECKING, NamedTuple
from os import path
from abc import ABC, abstractmethod
from ..db import DBWriter, DBReader
from ..scheduler.domain import Task, TaskQueue, Schedule
# To resolve circular import
if TYPE_CHECKING:
    from ..config import OKSPConfig

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
    def run(self, tasks: list[Task], job_name: str, config: 'OKSPConfig'):
        pass

    @abstractmethod
    def forecast(self, tasks: list[Task], config: 'OKSPConfig'):
        pass

    @staticmethod
    def serialize_tasks(schedule: Schedule, job_name: str, config: 'OKSPConfig'):
        dbpath = path.join(config.db_path, f'taskinfo-{job_name}.okspdb')
        with DBWriter(dbpath) as db:
            cc_db = db.create_collection('compute_configurations')
            alloc_db = db.create_collection('allocations')
            queue_db = db.create_collection('queues')
            task_db = db.create_collection('tasks')
            func_db = db.create_collection('functions')
            args_db = db.create_collection('args')
            kwargs_db = db.create_collection('kwargs')

            def add_queue(queue: TaskQueue):
                # TODO: Serialize child queues
                task_ids = []
                for task in queue.tasks:
                    func_id = func_db.insert(task.runnable.func)
                    arg_ids = [args_db.insert(arg) for arg in task.runnable.args]
                    kwarg_ids = [kwargs_db.insert(kwarg) for kwarg in task.runnable.kwargs.items()]
                    task_ids.append(task_db.insert(DBTask(func_id, arg_ids, kwarg_ids)))
                qid = queue_db.insert(DBQueue(
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
                    alloc_ids.append(alloc_db.insert(DBAllocation(queue_ids)))
                cc_db.insert(DBComputeConfiguration(alloc_ids))
        
        return dbpath

    @staticmethod
    def run_serialized_queue(dbpath: str, id: int):
        with DBReader(dbpath) as db:
            queue: DBQueue = db.get('queues', id)
            for task_id in queue.tasks:
                task: DBTask = db.get('tasks', task_id)
                func = db.get('functions', task.function)
                args = [db.get('args', arg_id) for arg_id in task.args]
                kwargs = {k:v for (k,v) in (db.get('kwargs', kwarg_id) for kwarg_id in task.kwargs)}
                func(*args, **kwargs)
