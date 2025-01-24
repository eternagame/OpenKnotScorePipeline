from typing import NamedTuple, Callable, Iterable
from os import path
from abc import ABC, abstractmethod
from ..taskdb import TaskDB
from ..scheduler.domain import Task, Schedule
from ..deferred import flush_deferred

class Runner(ABC):
    @abstractmethod
    def run(self, tasks: list[Task], job_name: str, on_queue: Callable[[Iterable[Task]], None] | None = None):
        pass

    @abstractmethod
    def forecast(self, tasks: list[Task], config):
        pass

    @staticmethod
    def serialize_tasks(schedule: Schedule, job_name: str, db_path: str):
        dbpath = path.join(db_path, f'taskinfo-{job_name}.db')
        with TaskDB(dbpath, 'n') as db:
            db.write_schedule(schedule)
        
        return dbpath

    @staticmethod
    def run_serialized_queue(dbpath: str, id: int):
        print('Running serialized queue', id)
        with TaskDB(dbpath) as db:
            for task in db.tasks_for_queue(id):
                task.run()
        flush_deferred()
