import sys
from os import path
sys.path.append(path.join(path.dirname(__file__), '../..'))

from openknotscore.cli import OKSPConfig
from openknotscore.runners.slurm import SlurmRunner

class Config(OKSPConfig):
    source_files = path.join(path.dirname(__file__), 'source_rdats/*')
    db_path = path.join(path.dirname(__file__), 'db')
    runner = SlurmRunner(
        partition='default',
        # You will only be able to allocate jobs on nodes with this many cores available
        # at the same time, so higher numbers can potentially queue slower
        max_cores=4,
        # Slurm limits this per partition via the MaxMemPerCPU QOS
        max_mem_per_core=8000,
        # Slurm limits the amount of queued jobs in a given partition via the MaxSubmitPU/MaxSubmitPA QOS,
        # and you may be further restricted to the actual resources that are requested by MaxTRESPU.
        # There may also be a separate limit configured for the number of jobs that can be submitted over some
        # time period. If you're submitting other jobs, make sure to leave room for them too!
        max_jobs=1000,
        # Slurm limits this this per partition  via the MaxWall QOS. Also, shorter jobs may be
        # easier to queue (and can just have the ability to finish faster)
        max_timeout=60*60*24,
        # Costs come from TRESBillingWeights
        cpu_cost=1,
        gpu_cost=10,
        # Memory cost is per GB
        mem_cost=0.25,
        # You will only be able to allocate jobs on nodes with this many GPUs available
        # at the same time, so higher numbers can potentially queue slower
        max_gpus=0,
        # Since we need to make sure we have enough memory available for the job to run,
        # we specify the amount of GPU memory that will be requested for GPUs
        gpu_memory=0,
        # If we were to require a GPU, we'd probably want to add a constraint to ensure
        # the GPU is of appropriate size. There may also be other requirements for things like
        # GPU architecture, CPU model for benchmarking, etc
        constraints=None
    )