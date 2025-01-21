import os
from os import path
import re
import math
from decimal import Decimal
from typing import Union, Callable, Iterable
from dataclasses import dataclass
import itertools
from datetime import datetime
import multiprocessing
from subprocess import run
from ..scheduler.domain import Schedule, ComputeConfiguration, Task
from ..scheduler.scheduler import schedule_tasks
from .runner import Runner, DBQueue
from ..taskdb import TaskDB, DBQueue

@dataclass
class SlurmRunner(Runner):
    db_path: str
    partitions: str
    max_cores: int
    max_mem_per_core: int
    max_jobs: int
    max_timeout: int
    cpu_cost: int | float
    gpu_cost: int | float
    mem_cost: int | float
    '''Per GiB'''
    max_gpus: int = 0
    gpu_memory: int = 0
    constraints: str = None
    init_script: str = None

    def _cost(self, cpus: int, gpus: int, memory: int, runtime: int):
        actual_cores = max(cpus, math.ceil(memory / self.max_mem_per_core))
        return Decimal((
            actual_cores * self.cpu_cost +
            gpus * self.gpu_cost +
            memory * self.mem_cost / 1024 / 1024 / 1024
        ) * runtime)

    def config_max_cpus(self, config: ComputeConfiguration):
        return max(
            max(
                sum(
                    max(task.utilized_resources.cpus for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
                ),
                math.ceil(
                    sum(
                        max(task.utilized_resources.memory for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
                    ) / self.max_mem_per_core
                )
            )
            for alloc in config.allocations
        )
    
    def config_max_memory(self, config: ComputeConfiguration):
        max(
            sum(
                max(task.utilized_resources.memory for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
            ) for alloc in config.allocations
        )

    def config_max_runtime(self, config: ComputeConfiguration):
        return max(
            max(
                sum(task.utilized_resources.max_runtime for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
            ) for alloc in config.allocations
        )
    
    def config_max_gpus(self, config: ComputeConfiguration):
        return max(
            len(
                set(queue.gpu_id for queue in alloc.queues if queue.gpu_id != None)
            ) for alloc in config.allocations
        )

    def run(self, tasks: list[Task], job_name: str, on_queue: Callable[[Iterable[Task]], None] | None = None):
        print(f'{datetime.now()} Scheduling tasks...')
        schedule: Schedule = schedule_tasks(
            tasks,
            [
                ComputeConfiguration(
                    self.max_cores,
                    self.max_cores * self.max_mem_per_core,
                    self.max_gpus,
                    self.max_timeout,
                    self.gpu_memory,
                )
            ]
        )
        
        print(f'{datetime.now()} Serializing tasks...')
        dbpath = self.serialize_tasks(schedule, job_name, self.db_path)
        
        print(f'{datetime.now()} Submitting batches...')
        allocated_jobs = 0
        for idx, comp_config in enumerate(schedule.nonempty_compute_configurations()):
            if allocated_jobs >= self.max_jobs:
                break

            allocations = comp_config.nonempty_allocations()
            array_size = min(len(allocations), self.max_jobs - len(allocations))
            allocated_jobs += array_size

            max_gpus = self.config_max_gpus(comp_config)
            cmds = [
                f'python -c "from openknotscore.substation.runners.slurm import SlurmRunner; SlurmRunner.run_serialized_allocation(\'{dbpath}\', {comp_config.id}, {'$SLURM_ARRAY_TASK_ID' if array_size > 1 else 0})"'
            ]
            if self.init_script: cmds.insert(0, self.init_script)
            sbatch(
                cmds,
                f'{job_name}-{idx}' if len(schedule.compute_configurations) > 1 else job_name,
                path.join(self.db_path, f'slurm-logs'),
                timeout=self.config_max_runtime(comp_config),
                partition=self.partitions,
                cpus=self.config_max_cpus(comp_config),
                gpus=max_gpus if max_gpus > 0 else None,
                memory_per_node=self.config_max_memory(comp_config),
                constraint=self.constraints,
                mail_type='END',
                array=f'0-{array_size-1}' if array_size > 1 else None,
                echo_cmd=True
            )
            
            if on_queue:
                on_queue(itertools.chain.from_iterable(alloc.tasks() for alloc in allocations[0:array_size]))


    def forecast(self, tasks):        
        schedule: Schedule = schedule_tasks(
            tasks,
            [
                ComputeConfiguration(
                    self.max_cores,
                    self.max_cores * self.max_mem_per_core,
                    self.max_gpus,
                    self.max_timeout,
                    self.gpu_memory,
                )
            ],
        )

        max_runtime = 0
        core_seconds = 0
        gpu_seconds = 0
        for task in schedule.tasks:
            max_runtime = max(task.utilized_resources.max_runtime, max_runtime)
            core_seconds += task.utilized_resources.max_runtime * task.utilized_resources.cpus
            if task.queue.gpu_id is not None:
                gpu_seconds += task.utilized_resources.max_runtime

        nonempty_configs = schedule.nonempty_compute_configurations()

        print(f'Slurm TRES cost: {sum(
            self._cost(config.cpus, config.gpus, config.memory, config.runtime) * len(config.nonempty_allocations())
            for config in nonempty_configs
        )}')
        print(f'Longest task runtime: {max_runtime / 60 / 60: .2f} hours ({max_runtime} seconds)')
        print(f'Active core-hours: {core_seconds / 60 / 60: .2f}')
        print(f'Allocated core-hours: {sum([
            self.config_max_cpus(config) * self.config_max_runtime(config) * len(config.nonempty_allocations())
            for config in nonempty_configs
        ]) / 60 / 60: .2f}')
        if gpu_seconds > 0:
            print(f'Active GPU-hours: {gpu_seconds / 60 / 60: .2f}')
            print(f'Allocated GPU-hours: {sum([
                self.config_max_gpus(config) * self.config_max_runtime(config) * len(config.nonempty_allocations())
                for config in nonempty_configs
            ]) / 60 / 60: .2f}')
        print(f'Number of jobs: {len(schedule.nonempty_compute_allocations())}')
        print(f'Longest job timeout: {max(config.runtime for config in nonempty_configs)} seconds')

    @staticmethod
    def _srun_queue(dbpath: str, queue: DBQueue, finished_queues: multiprocessing.Queue):
        srun(
            ['python', '-c', f'from openknotscore.substation.runners.runner import Runner; Runner.run_serialized_queue("{dbpath}", {queue.id})'],
            cpus=queue.cpus,
            gpu_cmode='shared' if queue.gpu_id != None else None,
            cuda_visible_devices=queue.gpu_id
        )
        finished_queues.put(queue)

    @staticmethod
    def run_serialized_allocation(dbpath: str, compute_config_id: int, allocation_id: int):
        with TaskDB(dbpath) as db:
            finished_queues = multiprocessing.Queue()
            running_queues = 0
            for queue in db.queues_for_allocation(compute_config_id, allocation_id):
                multiprocessing.Process(target=SlurmRunner._srun_queue, args=(dbpath, queue, finished_queues), daemon=True).start()
                running_queues += 1
            while running_queues > 0:
                finished: DBQueue = finished_queues.get()
                running_queues -= 1
                for queue in db.children_for_queue(finished.id):
                    multiprocessing.Process(target=SlurmRunner._srun_queue, args=(dbpath, queue, finished_queues), daemon=True).start()
                    running_queues += 1

def srun(
    command: list[str],
    cpus: int = None,
    gpu_cmode: str = None,
    cuda_visible_devices: str = None
):
    args = ['srun']
    environ = {**os.environ}

    if cpus is not None:
        args.append(f'--cpus-per-task={cpus}')
    
    if gpu_cmode is not None:
        args.append(f'--gpu_cmode={gpu_cmode}')

    if cuda_visible_devices is not None:
        environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    args += command

    if os.environ.get('SUBSTATION_RUNNER_DRY_RUN') == 'true':
        print(f'DRY RUN SRUN COMMAND: {args}\n')
        return 0
    else:
        run(args, input=input, text=True, cwd=os.getcwd())

def sbatch(
    commands: Union[str, list[str]],
    job_name: str,
    output_dir: str,
    timeout: str = None,
    partition: str = None,
    cpus: int = None,
    gpus: int = None,
    memory_per_node: str = None,
    memory_per_cpu: str = None,
    ntasks: int = None,
    dependency: str = None,
    mail_type: str = None,
    constraint: str = None,
    array: str = None,
    echo_cmd: bool = False,
):
    args = ['sbatch']

    if job_name is not None:
        args.append(f'--job-name={job_name}')
        job_id = '%j'
        if array is not None:
            job_id = '%A_%a'
        args.append(f'--output={output_dir}/{job_name}-{job_id}.%s.txt')
    
    if timeout is not None:
        args.append(f'--time={timeout}')

    if partition is not None:
        args.append(f'--partition={partition}')

    if cpus is not None:
        args.append(f'--cpus-per-task={cpus}')
    
    if gpus is not None:
        args.append(f'--gpus={gpus}')

    if memory_per_node is not None:
        args.append(f'--mem={memory_per_node}')
    
    if memory_per_cpu is not None:
        args.append(f'--mem-per-cpu={memory_per_cpu}')

    if ntasks is not None:
        args.append(f'--ntasks={ntasks}')

    if dependency is not None:
        args.append(f'--dependency={dependency}')
    
    if mail_type is not None:
        args.append(f'--mail-type={mail_type}')
    
    if constraint is not None:
        args.append(f'--constraint={constraint}')

    if array is not None:
        args.append(f'--array={array}')

    shebang = '#!/bin/sh'
    set = 'set -e'
    if echo_cmd:
        set += 'x'
    body = (commands if isinstance(commands, str) else '\n'.join(commands))

    input = f'{shebang}\n\n{set}\n\n{body}'

    if os.environ.get('SUBSTATION_RUNNER_DRY_RUN') == 'true':
        print(f'DRY RUN SBATCH STDIN:\n{input}\n-------')
        print(f'DRY RUN SBATCH COMMAND: {args}\n--------------')
    else:
        res = run(args, input=input, text=True, cwd=os.getcwd(), capture_output=True)
        # Return job ID of queued job
        match = re.search(r'Submitted batch job (\d+)', res.stdout)
        if match is None:
            raise Exception(f'sbatch did not return job id.\nstdout\n{res.stdout}\nstderr\n{res.stderr}')
        return match.group(1)
