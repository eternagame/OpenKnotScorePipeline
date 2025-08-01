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
from .runner import Runner
from ..taskdb import TaskDB, DBQueue

# We'll reserve an extra 10MB in overhead per task that isnt reserved for any queue
# to account for the base script and rounding errors to ensure srun has enough memory
# to allocate for all steps
MEM_BUFFER = 10*1024**2

@dataclass
class SlurmConfig:
    partitions: str
    max_cores: int
    max_mem_per_core: int
    max_timeout: int
    max_gpus: int = 0
    gpu_memory: int = 0
    constraints: str = None
    init_script: str = None
    runtime_buffer: int = 0
    max_concurrent: int = None

@dataclass
class SlurmRunner(Runner):
    db_path: str
    max_jobs: int
    cpu_cost: int | float
    gpu_cost: int | float
    mem_cost: int | float
    '''Per GiB'''
    configs: list[SlurmConfig]

    def _cost(self, cpus: int, gpus: int, memory: int, runtime: int, max_mem_per_core: int):
        actual_cores = max(cpus, math.ceil(memory / max_mem_per_core))
        return Decimal((
            actual_cores * self.cpu_cost +
            gpus * self.gpu_cost +
            memory * self.mem_cost / 1024 / 1024 / 1024
        ) * runtime)

    def config_max_cpus(self, config: ComputeConfiguration, max_mem_per_core: int, with_memory_overage=True):
        return max(
            max(
                sum(
                    max(task.utilized_resources.cpus for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
                ),
                # When eg calculating cost, we want to consider the fact that we will actually be allocating extra CPUs if we use more
                # memory per core than the max memory per core setting. However, when we actually go to request the resources,
                # we don't want to *explicitly* ask for those extra cores, as the actual Slurm configuration (vs what the user passed to the runner)
                # may not require it
                math.ceil(
                    sum(
                        max(task.utilized_resources.memory for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
                    ) / max_mem_per_core
                ) if with_memory_overage else 0
            )
            for alloc in config.allocations
        )
    
    def config_max_memory(self, config: ComputeConfiguration):
        return max(
            sum(
                max(task.utilized_resources.memory for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
            ) for alloc in config.allocations
        ) + MEM_BUFFER

    def config_max_runtime(self, config: ComputeConfiguration, buffer: int):
        return max(
            max(
                sum(task.utilized_resources.max_runtime for task in queue.tasks) if len(queue.tasks) > 0 else 0 for queue in alloc.queues
            ) for alloc in config.allocations
        ) + buffer
    
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
                    config.max_cores,
                    config.max_cores * config.max_mem_per_core - MEM_BUFFER,
                    config.max_gpus,
                    config.max_timeout - config.runtime_buffer,
                    config.gpu_memory,
                )
                for config in self.configs
            ],
            True
        )
        
        print(f'{datetime.now()} Serializing tasks...')
        dbpath = self.serialize_tasks(schedule, job_name, self.db_path)
        
        print(f'{datetime.now()} Submitting batches...')
        allocated_jobs = 0
        for idx, comp_config in enumerate(schedule.nonempty_compute_configurations()):
            slurm_config = self.configs[schedule.compute_configurations.index(comp_config)]

            if allocated_jobs >= self.max_jobs:
                break

            allocations = comp_config.nonempty_allocations()
            array_size = min(len(allocations), self.max_jobs - allocated_jobs)
            allocated_jobs += array_size

            max_gpus = self.config_max_gpus(comp_config)
            cmds = [
                f'python -c "from openknotscore.substation.runners.slurm import SlurmRunner; SlurmRunner.run_serialized_allocation(\'{dbpath}\', {comp_config.id}, {'$SLURM_ARRAY_TASK_ID' if array_size > 1 else 0})"'
            ]
            init_script = slurm_config.init_script
            if init_script: cmds.insert(0, init_script)

            max_runtime = self.config_max_runtime(comp_config, slurm_config.runtime_buffer)

            if on_queue:
                on_queue(itertools.chain.from_iterable(alloc.tasks() for alloc in allocations[0:array_size]))

            sbatch(
                cmds,
                f'{job_name}-{idx+1}' if len(schedule.nonempty_compute_configurations()) > 1 else job_name,
                path.join(self.db_path, f'slurm-logs'),
                timeout=f'{max_runtime // 60}:{max_runtime % 60}',
                partition=slurm_config.partitions,
                cpus=self.config_max_cpus(comp_config, slurm_config.max_mem_per_core, with_memory_overage=False),
                gpus=max_gpus if max_gpus > 0 else None,
                memory_per_node=f'{self.config_max_memory(comp_config)//1024}K',
                constraint=slurm_config.constraints,
                mail_type='END',
                array=f'0-{array_size-1}{('%' + str(slurm_config.max_concurrent)) if slurm_config.max_concurrent else ''}' if array_size > 1 else None,
                echo_cmd=True,
                nodes=1,
            )

    def forecast(self, tasks):        
        schedule: Schedule = schedule_tasks(
            tasks,
            [
                ComputeConfiguration(
                    config.max_cores,
                    config.max_cores * config.max_mem_per_core,
                    config.max_gpus,
                    config.max_timeout - config.runtime_buffer,
                    config.gpu_memory,
                )
                for config in self.configs
            ],
            True
        )

        max_runtime = 0
        max_core_seconds = 0
        max_gpu_seconds = 0
        max_expected_runtime = 0
        expected_core_seconds = 0
        expected_gpu_seconds = 0
        for task in schedule.tasks:
            max_runtime = max(task.utilized_resources.max_runtime, max_runtime)
            max_expected_runtime = max(task.utilized_resources.avg_runtime, max_expected_runtime)
            max_core_seconds += task.utilized_resources.max_runtime * task.utilized_resources.cpus
            expected_core_seconds += task.utilized_resources.avg_runtime * task.utilized_resources.cpus
            if task.queue.gpu_id is not None:
                max_gpu_seconds += task.utilized_resources.max_runtime
                expected_gpu_seconds += task.utilized_resources.avg_runtime

        nonempty_configs = schedule.nonempty_compute_configurations()
        nonempty_allocations = schedule.nonempty_compute_allocations()

        print(f'Slurm TRES cost: {sum(
            self._cost(config.cpus, config.gpus, config.memory, config.runtime, self.configs[schedule.compute_configurations.index(config)].max_mem_per_core) * len(config.nonempty_allocations())
            for config in nonempty_configs
        )}')
        print(f'Longest task runtime (max): {max_runtime / 60 / 60: .2f} hours ({max_runtime} seconds)')
        print(f'Longest task runtime (expected): {max_expected_runtime / 60 / 60: .2f} hours ({max_expected_runtime} seconds)')
        print(f'Active core-hours (max): {max_core_seconds / 60 / 60: .2f}')
        print(f'Active core-hours (expected): {expected_core_seconds / 60 / 60: .2f}')
        print(f'Allocated core-hours: {sum([
            self.config_max_cpus(config, self.configs[schedule.compute_configurations.index(config)].max_mem_per_core) * self.config_max_runtime(config, self.configs[schedule.compute_configurations.index(config)].runtime_buffer) * len(config.nonempty_allocations())
            for config in nonempty_configs
        ]) / 60 / 60: .2f}')
        if max_gpu_seconds > 0:
            print(f'Active GPU-hours (max): {max_gpu_seconds / 60 / 60: .2f}')
            print(f'Active GPU-hours (expected): {expected_gpu_seconds / 60 / 60: .2f}')
            print(f'Allocated GPU-hours: {sum([
                self.config_max_gpus(config) * self.config_max_runtime(config, self.configs[schedule.compute_configurations.index(config)].runtime_buffer) * len(config.nonempty_allocations())
                for config in nonempty_configs
            ]) / 60 / 60: .2f}')
        print(f'Number of jobs: {len(nonempty_allocations)}')
        longest_job_timeout = max(self.config_max_runtime(config, self.configs[schedule.compute_configurations.index(config)].runtime_buffer) for config in nonempty_configs)
        print(f'Longest job timeout: {longest_job_timeout / 60 / 60: .2f} hours ({longest_job_timeout} seconds)')
        longest_job_expected = max(alloc.utilized_resources.avg_runtime for alloc in nonempty_allocations)
        print(f'Longest job runtime (expected): {longest_job_expected / 60 / 60: .2f} hours ({longest_job_expected} seconds)')
        avg_job_max = sum(alloc.utilized_resources.max_runtime for alloc in nonempty_allocations) / len(nonempty_allocations)
        print(f'Average job runtime (max): {avg_job_max / 60 / 60: .2f} hours ({avg_job_max} seconds)')
        avg_job_expected = sum(alloc.utilized_resources.avg_runtime for alloc in nonempty_allocations) / len(nonempty_allocations)
        print(f'Average job runtime (expected): {avg_job_expected / 60 / 60: .2f} hours ({avg_job_expected} seconds)')

    @staticmethod
    def _srun_queue(dbpath: str, queue: DBQueue, finished_queues: multiprocessing.Queue):
        srun(
            ['/usr/bin/env', 'python', '-c', f'from openknotscore.substation.runners.runner import Runner; Runner.run_serialized_queue("{dbpath}", {queue.id})'],
            cpus=queue.cpus,
            memory_per_node=f'{math.ceil(queue.memory/1024)}K',
            gpus=1 if queue.gpu_id != None else None,
            export='ALL'
        )
        finished_queues.put(queue)

    @staticmethod
    def run_serialized_allocation(dbpath: str, compute_config_id: int, allocation_id: int):
        with TaskDB(dbpath) as db:
            finished_queues = multiprocessing.Queue()
            running_queues = []
            for queue in db.queues_for_allocation(compute_config_id, allocation_id):
                print('Triggering srun for base queue', queue.id)
                p = multiprocessing.Process(target=SlurmRunner._srun_queue, args=(dbpath, queue, finished_queues), daemon=True)
                p.start()
                running_queues.append({'proc': p, 'queue': queue})
            while len(running_queues) > 0:
                try:
                    finished_queues.get(timeout=30)
                except:
                    pass
                
                finished = next((queue for queue in running_queues if not queue['proc'].is_alive()), None)
                if finished:
                    running_queues.remove(finished)
                    for queue in db.children_for_queue(finished['queue'].id):
                        print('Triggering srun for queue', queue.id, 'child of', finished['queue'].id)
                        p = multiprocessing.Process(target=SlurmRunner._srun_queue, args=(dbpath, queue, finished_queues), daemon=True)
                        p.start()
                        running_queues.append({'proc': p, 'queue': queue})

def srun(
    command: list[str],
    cpus: int = None,
    memory_per_node: str = None,
    memory_per_cpu: str = None,
    gpus: int = None,
    gpu_cmode: str = None,
    cuda_visible_devices: str = None,
    export: str = None
):
    args = ['srun']
    environ = {**os.environ}

    args.append('-n1')

    if 'SUBSTATION_BATCH_OUTPUT_PATTERN' in os.environ:
        args.append(f'--output={os.environ["SUBSTATION_BATCH_OUTPUT_PATTERN"]}')

    if cpus is not None:
        args.append(f'--cpus-per-task={cpus}')

    if memory_per_node is not None:
        args.append(f'--mem={memory_per_node}')
    
    if memory_per_cpu is not None:
        args.append(f'--mem-per-cpu={memory_per_cpu}')
    
    if gpus is not None:
        args.append(f'--gpus={gpus}')

    if gpu_cmode is not None:
        args.append(f'--gpu_cmode={gpu_cmode}')

    if export is not None:
        args.append(f'--export={export}')

    if cuda_visible_devices is not None:
        environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    args += command

    if os.environ.get('SUBSTATION_RUNNER_DRY_RUN') == 'true':
        print(f'DRY RUN SRUN COMMAND: {args}\n')
        return 0
    else:
        run(args, text=True, cwd=os.getcwd(), env=environ)

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
    nodes: int = None
):
    args = ['sbatch']

    output_pattern = ''
    if job_name is not None:
        args.append(f'--job-name={job_name}')
        job_id = '%j'
        if array is not None:
            job_id = '%A_%a'
        output_pattern = f'{output_dir}/{job_name}-{job_id}.%s.txt'
        args.append(f'--output={output_pattern}')
    
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

    if nodes is not None:
        args.append(f'--nodes={nodes}')

    if dependency is not None:
        args.append(f'--dependency={dependency}')
    
    if mail_type is not None:
        args.append(f'--mail-type={mail_type}')
    
    if constraint is not None:
        args.append(f'--constraint={constraint}')

    if array is not None:
        args.append(f'--array={array}')

    shebang = '#!/bin/sh'
    setsh = 'set -e'
    if echo_cmd:
        setsh += 'x'
    export_outpattern = ''
    if output_pattern:
        export_outpattern = f'export SUBSTATION_BATCH_OUTPUT_PATTERN={output_pattern}\n\n'
    body = (commands if isinstance(commands, str) else '\n'.join(commands))

    input = f'{shebang}\n\n{setsh}\n\n{export_outpattern}{body}'

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
