from typing import NamedTuple, Callable, Iterable
from os import path
from abc import ABC, abstractmethod
from ..taskdb import TaskDB
from ..scheduler.domain import Task, Schedule
from ..deferred import flush_deferred

class Runner(ABC):
    @abstractmethod
    def run(self, tasks: list[Task], job_name: str, on_queue: Callable[[Iterable[Task]], None] | None = None):
        '''
        Runs a set of task using this runner

        tasks: the list of tasks (defining functions to be run and their required resources)
        job_name: A human-friendly way to refer to the entire set of work to be done,
            which can eg be used by SlurmRunner to set the slurm job names of submitted jobs
        on_queue: Once a batch of tasks have been queued (eg selected to run locally,
            submitted as part of a slurm job, etc) this function will be called with those
            tasks so that eg you can mark them as such in your local application
        '''
        pass

    @abstractmethod
    def forecast(self, tasks: list[Task]):
        '''
        Prints information about anticipated resources required to complete the given tasks
        '''
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
