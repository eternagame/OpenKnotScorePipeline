from typing import Callable, Iterable
import os
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import dataclass
from ..scheduler.domain import Task
from .runner import Runner
from ..deferred import flush_deferred

# https://github.com/python/cpython/issues/111873#issuecomment-1809135036
def start_orphan_checker():
    import threading

    def exit_if_orphaned():
        import multiprocessing
        multiprocessing.parent_process().join()  # wait for parent process to die first; may never happen
        os._exit(-1)

    threading.Thread(target=exit_if_orphaned, daemon=True).start()

@dataclass
class LocalRunner(Runner):
    max_processes: int = cpu_count() - 1

    def _run(self, f):
        f()

    def run(self, tasks: list[Task], job_name: str, on_queue: Callable[[Iterable[Task]], None] | None = None):
        if on_queue:
            on_queue(tasks)

        if os.environ.get('SUBSTATION_RUNNER_DRY_RUN') == 'true':
            print('DRY RUN LOCAL')
            return
        
        with Pool(self.max_processes, initializer=start_orphan_checker) as pool:
            # Iterate over results so that we catch errors
            for _ in pool.map(self._run, [task.runnable.run for task in tasks]):
                pass

        flush_deferred()

    def forecast(self, tasks):
        # This is all very unscientific (tasks may utilize an arbitrary number of CPUs,
        # we don't do any checks around having sufficient RAM or VRAM, etc).
        # LocalRunner is not really expected to be used for serious use cases anyways vs testing
        # and quick jobs. Someone may want to improve this later
        total_time = sum(task.utilized_resources.max_runtime for task in tasks)
        print(f'Total core-hours: {total_time/60/60: .2f}')
        print(f'Execution time: {total_time/60/60/self.max_processes: .2f} hours')
