import os
import subprocess
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import dataclass
from ..plan.domain import AccessibleResources, Task
from .runner import Runner

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

    def run(self, tasks, job_name, config):
        if os.environ.get('OKSP_RUNNER_DRY_RUN') == 'true':
            print('DRY RUN LOCAL')
            return
        
        with Pool(self.max_processes, initializer=start_orphan_checker) as pool:
            # Iterate over results so that we catch errors
            for _ in pool.map(self._run, [task.runnable.run for task in tasks]):
                pass

    def forecast(self, tasks, config):
        # This is all very unscientific (tasks may utilize more than one CPU if available,
        # we don't do any checks around having sufficient RAM or VRAM, etc).
        # LocalRunner is not really expected to be used for serious use cases anyways vs testing
        # and quick jobs. Someone may want to improve this later
        try:
            subprocess.check_output('nvidia-smi')
            gpu_available = True
        except:
            gpu_available = False

        total_time = sum(task.compute_utilized_resources(AccessibleResources(1, gpu_available)).max_runtime for task in tasks)
        print(f'Total core-hours: {total_time/60/60: .2f}')
        print(f'Execution time: {total_time/60/60/self.max_processes: .2f} hours')
