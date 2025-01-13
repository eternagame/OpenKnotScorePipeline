import argparse
import importlib
import os
import pandas as pd
from .pipeline.import_source import load_sources
from .pipeline.prediction.sample import generate_predictor_resource_model, MODEL_TIMEOUT
from .scheduler.domain import Task, Runnable, UtilizedResources
from .config import OKSPConfig

def run_cli():
    # NOTE: Any configuration needed to ensure reproducibility should be provided
    # via the config argument/class, NOT command line flags!
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='module', help='python module path containing Config class')

    subparsers = parser.add_subparsers(dest='cmd', help='subcommand', required=True)
    subparsers.add_parser('predict', help='generating missing structure predictions')
    subparsers.add_parser('score', help='compute scores using previously generated predictions and export to the final output files')
    subparsers.add_parser('predict-forecast', help='compute the total number of core-hours required for computation and minimum per-job requirements')
    model_parser = subparsers.add_parser('predict-generate-model', help='run predictors with sample inputs and generate resource usage models')
    model_parser.add_argument('--cpus', dest='cpus', help='cpus to allocate per predictor, in MB (limits not supported for local runner)', default=1)
    model_parser.add_argument('--max-memory', dest='max_memory', help='maximum memory to allocate per predictor, in MB (limits not supported for local runner)', default=1024*4)
    model_parser.add_argument('--max-gpu-memory', dest='max_gpu_memory', help='maximum GPU memory to allocate per predictor if it supports GPU, in MB (limits not supported for local runner)', default=0)

    args = parser.parse_args()

    config: OKSPConfig = importlib.import_module(args.module).Config
    os.makedirs(config.db_path, exist_ok=True)

    if args.cmd == 'predict-generate-model':
        config.runner.run(
            [
                Task(
                    Runnable.create(generate_predictor_resource_model)(predictor),
                    UtilizedResources(
                        MODEL_TIMEOUT, MODEL_TIMEOUT, MODEL_TIMEOUT,
                        args.cpus,
                        args.max_memory * 1024 * 1024,
                        args.max_gpu_memory if predictor.gpu else 0
                    )
                )
                for predictor in config.enabled_predictors
            ],
            'oksp-predict-generate-model',
            config
        )
    else:
        source_data: pd.DataFrame = config.filter_for_computation(load_sources(config.source_files))

        if args.cmd == 'predict-forecast' or args.cmd == 'predict':
            pred_tasks = [
                Task(
                    Runnable.create(predictor.run)(row['sequence'], row.get('reactivity')),
                    UtilizedResources(
                        predictor.approximate_max_runtime(row['sequence']),
                        predictor.approximate_avg_runtime(row['sequence']),
                        predictor.approximate_min_runtime(row['sequence']),
                        predictor.cpus,
                        predictor.approximate_max_memory(row['sequence']),
                        predictor.approximate_max_gpu_memory(row['sequence']),
                    )
                )
                for predictor in config.enabled_predictors
                for _, row in source_data.iterrows()
                if not predictor.uses_experimental_reactivities or 'reactivity' in row
            ]

            if args.cmd == 'predict-forecast':
                config.runner.forecast(pred_tasks, config)
            else:
                config.runner.run(
                    pred_tasks,
                    'oksp-predict',
                    config
                )
    
if __name__ == '__main__':
    run_cli()
