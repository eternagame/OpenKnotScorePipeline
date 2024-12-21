from abc import ABC, abstractmethod
import argparse
import pandas as pd
import os
from .pipeline.import_source import load_sources
from .pipeline.prediction import predictors
from .pipeline.prediction.prediction import get_prediction_task
from .plan.domain import Task
from .plan.solver import solve

class OKSPConfig(ABC):
    @property
    @abstractmethod
    def source_files() -> str | list[str]:
        '''
        A single or list of file paths or globs that should be loaded as input data to the pipeline
        '''
        pass

    @property
    @abstractmethod
    def db_path() -> str | list[str]:
        '''
        Path to a directory that can be used to store internal files (eg, caches, intermediate outputs, etc)
        '''
        pass

    def filter_for_computation(df: pd.DataFrame) -> pd.DataFrame:
        '''
        If you don't want to run computations for all data in source_files, use this function to
        filter down to just the rows you want to compute for
        '''
        return df
    
    enabled_predictors: list[predictors.Predictor] = [
        predictors.Vienna2Predictor(as_name='vienna2'),
        predictors.Contrafold2Predictor(
            as_name='contrafold2',
            arnie_kwargs={
                'params_file': os.environ['CONTRAFOLD_2_PARAMS_PATH']
            }
        ),
        predictors.EternafoldPredictor(as_name='eternafold'),
        predictors.RnastructurePredictor(as_name='rnastructure_mfe'),
        # Takes too long
        #predictors.E2efoldPredictor(as_name='e2efold'),
        predictors.HotknotsPredictor(as_name='hotknots'),
        predictors.IpknotsPredictor(as_name='ipknots'),
        predictors.KnottyPredictor(as_name='knotty'),
        predictors.PknotsPredictor(as_name='pknots'),
        predictors.SpotrnaPredictor(as_name='spotrna'),
        # Never got this working
        #predictors.Spotrna2Predictor(as_name='spotrna2')
        predictors.ShapifyHfoldPredictor(as_name='shapify-hfold', hfold_location=os.environ['HFOLD_PATH']),
        predictors.NupackPkPredictor(as_name='nupack_pk'),
        predictors.RnastructureShapePredictor(as_name='rnastructure+SHAPE'),
        predictors.ShapeknotsPredictor(as_name='shapeknots'),
        predictors.Vienna2PkFromBppPredictor()
            .add_heuristic('threshknots', as_name='vienna_2.TK')
            .add_heuristic('hungarian', as_name='vienna_2.HN'),
        predictors.EternafoldPkFromBppPredictor()
            .add_heuristic('threshknots', as_name='eternafold.TK')
            .add_heuristic('hungarian', as_name='eternafold.HN'),
        predictors.Contrafold2PkFromBppPredictor(arnie_bpp_kwargs={
                'params_file': os.environ['CONTRAFOLD_2_PARAMS_PATH']
            })
            .add_heuristic('threshknots', as_name='contrafold_2.TK')
            .add_heuristic('hungarian', as_name='contrafold_2.HN')
        ]

def run_cli(config: OKSPConfig):
    # NOTE: Any configuration needed to ensure reproducibility should be provided
    # via the config argument/class, NOT command line flags!
    parser = argparse.ArgumentParser()

    runner_parser = argparse.ArgumentParser(add_help=False)
    runner_parser.add_argument('--runner', dest='runner', choices=['local', 'slurm'], default='local', help='how to run computations (default: local)')
    runner_parser.add_argument('--max-cores', dest='max_cores', type=int, help='the maximum number of cores per runner job', required=True)
    runner_parser.add_argument('--max-mem-per-core', dest='max_mem_per_core', type=int, help='the maximum amount of memory available per core in mb', required=True)
    runner_parser.add_argument('--max-timeout', dest='max_timeout', type=int, help='the maximum timeout per runner job in seconds', required=True)
    runner_parser.add_argument('--max-jobs', dest='max_jobs', type=int, help='the maximum number of runner jobs to queue (tasks will be attempted to pack into this number of jobs, but if they don\'t fit this will mean multiple runs are needed)', required=True)
    runner_parser.add_argument('--cpu-cost', dest='cpu_cost', type=int, help='when computing optimal job packing, the relative cost associated with one cpu-hour', required=True)
    runner_parser.add_argument('--gpu-cost', dest='gpu_cost', type=int, help='when computing optimal job packing, the relative cost associated with one gpu-hour', required=True)
    runner_parser.add_argument('--mem-cost', dest='mem_cost', type=int, help='when computing optimal job packing, the relative cost associated with one mb-hour of RAM', required=True)
    runner_parser.add_argument('--allow-gpu', dest='allow_gpu', type=int, help='whether or not jobs should be run on a GPU if found to be optimal', required=True)
    runner_parser.add_argument('--slurm-partitions', dest='slurm_partitions', type=int, help='if runner is slurm, the partitions to queue into')

    subparsers = parser.add_subparsers(dest='cmd', help='subcommand', required=True)
    subparsers.add_parser('forecast', help='compute the total number of core-hours required for computation and minimum per-job requirements')
    subparsers.add_parser('all', help='do a one-shot pipeline run - (equivilent to score + predict)', parents=[runner_parser])
    subparsers.add_parser('predict', help='generating missing structure predictions', parents=[runner_parser])
    subparsers.add_parser('score', help='compute scores using previously generated predictions and export to the final output files', parents=[runner_parser])

    args = parser.parse_args()

    if args.cmd in ['forecast', 'all', 'predict', 'score']:
        source_data: pd.DataFrame = config.filter_for_computation(load_sources(config.source_files))

        if args.cmd in ['forecast', 'all', 'predict']:
            pred_tasks = [
                get_prediction_task(predictor, row)
                for predictor in config.enabled_predictors
                for idx, row in source_data.iterrows()
            ]
            pred_schedule = solve(
                pred_tasks,
                args.slurm_max_jobs if args.runner == 'slurm' else 1,
                args.max_cores,
                args.cpu_cost,
                args.allow_gpu,
                args.gpu_cost,
                args.max_mem_per_core,
                args.mem_cost,
                args.max_timeout
            )

        if args.cmd == 'forecast':
            max_runtime = 0
            core_seconds = 0
            gpu_seconds = 0
            for task in pred_schedule.tasks:
                runtime = task.calc_max_utilized_runtime(task.shard.resources) 
                max_runtime = max(runtime, max_runtime)
                core_seconds += runtime * task.shard.resources.cpus
                if task.shard.resources.gpu:
                    gpu_seconds += runtime
            
            print('-------------------')
            print('Execution Plan')
            print('-------------------')
            print(f'Optimization score: {pred_schedule.score.soft_score}')
            print(f'Longest task runtime: {max_runtime / 60 / 60} hours ({max_runtime} seconds)')
            print(f'Active core-hours: {core_seconds / 60 / 60}')
            print(f'Allocated core-hours: {sum([
                alloc.allocated_cores() * alloc.requested_timeout for alloc in pred_schedule.compute_allocations
            ]) / 60 / 60}')
            print(f'Number of jobs: {len(alloc for alloc in pred_schedule.compute_allocations if any(shard for shard in alloc.shards if len(shard.tasks > 0)))}')
            print(f'Longest job timeout: {max(alloc.requested_timeout for alloc in pred_schedule.compute_allocations)}')
