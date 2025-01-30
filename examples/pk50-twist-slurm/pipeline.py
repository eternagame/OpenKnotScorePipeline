import sys
from os import path

from openknotscore.config import OKSPConfig, RDATOutput, CSVOutput, ParquetOutput
from openknotscore.substation.runners.slurm import SlurmRunner

class Config(OKSPConfig):
    db_path = path.join(path.dirname(__file__), 'db')

    source_files = path.join(path.dirname(__file__), 'source_rdats/*')
    
    # When calculating the expected max runtime needed to run tasks, use this multiplier to add
    # some additional buffer to account for underestimates or variation
    # runtime_buffer = 1.15
    # When calculating the expected RAM needed to run tasks, use this multiplier to add
    # some additional buffer to account for underestimates or variation
    # memory_buffer = 1.15
    # When calculating the expected GPU memory needed to run tasks, use this multiplier to add
    # some additional buffer to account for underestimates or variation
    # gpu_memory_buffer = 1.15
    runner = SlurmRunner(
        # Slurm limits the amount of queued jobs in a given partition via the MaxSubmitPU/MaxSubmitPA QOS,
        # and you may be further restricted to the actual resources that are requested by MaxTRESPU.
        # There may also be a separate limit configured for the number of jobs that can be submitted over some
        # time period. If you're submitting other jobs, make sure to leave room for them too!
        max_jobs=800,
        # Costs come from TRESBillingWeights
        cpu_cost=1,
        gpu_cost=10,
        # Memory cost is per GB
        mem_cost=0.25,
        configs=[
            SlurmConfig(
                partitions='default',
                # You will only be able to allocate jobs on nodes with this many cores available
                # at the same time, so higher numbers can potentially queue slower
                max_cores=12,
                # Slurm limits this per partition via the MaxMemPerCPU QOS
                max_mem_per_core=8000 * 1024 ** 2,
                # Slurm limits this this per partition  via the MaxWall QOS. Also, shorter jobs may be
                # easier to queue (and can just have the ability to finish faster)
                max_timeout=60*60*24,
                # Pass extra content which should be run at the start of sbatch scripts, eg to load
                # packages or set environment variables
                # init_script='',
                # Make sure the runtime of jobs has this many extra seconds free eg to run the init_script
                # runtime_buffer=0,
            ),
            # We set this as the second one as it is more expensive - we'd prefer not to use it unless
            # we have to!
            SlurmConfig(
                partitions='gpu',
                max_cores=12,
                max_mem_per_core=8000 * 1024 ** 2,
                max_timeout=60*60*24,
                # You will only be able to allocate jobs on nodes with this many GPUs available
                # at the same time, so higher numbers can potentially queue slower
                max_gpus=1,
                # Since we need to make sure we have enough memory available for the job to run,
                # we specify the amount of GPU memory that will be requested for GPUs
                gpu_memory=24000*1024**2,
                # Ensure the GPU is of appropriate size. You could also add requirements for things like
                # GPU architecture, CPU model for benchmarking, etc
                constraints='GPU_MEM:24GB',
                # init_script='',
                # rutime_buffer=0
            )
        ]
    )

    output_configs = [
        RDATOutput(path.join(path.dirname(__file__), 'output/rdat')),
        RDATOutput(path.join(path.dirname(__file__), 'output/rdat-eterna'), 100),
        CSVOutput(path.join(path.dirname(__file__), 'output/structures.csv')),
        ParquetOutput(path.join(path.dirname(__file__), 'output/structures.parquet.gz'), 'gzip'),
    ]
