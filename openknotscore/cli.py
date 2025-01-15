import argparse
import importlib
import os
from os import path
from datetime import datetime
import pandas as pd
from .pipeline.import_source import load_sources
from .pipeline.prediction.sample import generate_predictor_resource_model, MODEL_TIMEOUT
from .pipeline.prediction.predict import predict, PredictionDB
from .substation.scheduler.domain import Task, Runnable, UtilizedResources
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
            'oksp-predict-generate-model'
        )
    else:
        print(f'{datetime.now()} Loading data...')
        source_data: pd.DataFrame = config.filter_for_computation(load_sources(config.source_files))

        pred_db_path = path.join(config.db_path, 'predictions.db')
        if args.cmd == 'predict-forecast' or args.cmd == 'predict':
            print(f'{datetime.now()} Generating tasks...')
            pred_tasks = [
                Task(
                    Runnable.create(predict)(predictor, row['sequence'], row.get('reactivity'), pred_db_path),
                    predictor.approximate_resources(row['sequence'])
                )
                for predictor in config.enabled_predictors
                for _, row in source_data.iterrows()
                if not predictor.uses_experimental_reactivities or 'reactivity' in row
            ]

            if args.cmd == 'predict-forecast':
                print(f'{datetime.now()} Starting forecast...')
                config.runner.forecast(pred_tasks)
            else:
                print(f'{datetime.now()} Starting run...')
                with PredictionDB(pred_db_path, 'c') as preddb:
                    def on_queued(tasks: list[Task]):
                        for task in tasks:
                            for name in task.runnable.args[0].prediction_names:
                                preddb.set_queued(name, task.runnable.args[1], task.runnable.args[2])
                    config.runner.run(pred_tasks, 'oksp-predict', on_queued)
                print(f'{datetime.now()} Completed')
    
if __name__ == '__main__':
    run_cli()
