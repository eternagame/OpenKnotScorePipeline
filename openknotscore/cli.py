import argparse
import importlib
import os
from os import path
from datetime import datetime
import itertools
import pandas as pd
from typing import Iterable
from .pipeline.import_source import load_sources
from .pipeline.prediction.sample import generate_predictor_resource_model, MODEL_TIMEOUT
from .pipeline.prediction.predict import predict, PredictionDB, PredictionStatus
from .substation.scheduler.domain import Task, Runnable, UtilizedResources
from .config import OKSPConfig

def run_cli():
    # NOTE: Any configuration needed to ensure reproducibility should be provided
    # via the config argument/class, NOT command line flags!
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='module', help='python module path containing Config class')

    subparsers = parser.add_subparsers(dest='cmd', help='subcommand', required=True)
    subparsers.add_parser('predict', help='generate missing structure predictions')
    subparsers.add_parser('score', help='compute scores using previously generated predictions and export to the final output files')
    subparsers.add_parser('predict-forecast', help='compute the total number of core-hours required for computation and minimum per-job requirements')
    model_parser = subparsers.add_parser('predict-generate-model', help='run predictors with sample inputs and generate resource usage models')
    model_parser.add_argument('--cpus', dest='cpus', help='cpus to allocate per predictor, in MB (limits not supported for local runner)', default=1)
    model_parser.add_argument('--max-memory', dest='max_memory', help='maximum memory to allocate per predictor, in MB (limits not supported for local runner)', default=1024*4)
    model_parser.add_argument('--max-gpu-memory', dest='max_gpu_memory', help='maximum GPU memory to allocate per predictor if it supports GPU, in MB (limits not supported for local runner)', default=0)
    import_parser = subparsers.add_parser('predict-import', help='import pre-computed structures from another file into the pipeline database')
    import_parser.add_argument(dest='file', help='path to the file to import structures from')
    import_parser.add_argument('--override', dest='override', help='override values in the database if they already exist', default=False)

    args = parser.parse_args()

    config: OKSPConfig = importlib.import_module(args.module).Config
    os.makedirs(config.db_path, exist_ok=True)

    pred_db_path = path.join(config.db_path, 'predictions.db')
    if args.cmd == 'predict-generate-model':
        config.runner.run(
            [
                Task(
                    Runnable.create(generate_predictor_resource_model)(predictor),
                    UtilizedResources(
                        MODEL_TIMEOUT * config.runtime_buffer, MODEL_TIMEOUT, MODEL_TIMEOUT,
                        args.cpus,
                        args.max_memory * 1024 * 1024 * config.memory_buffer,
                        args.max_gpu_memory * config.gpu_memory_buffer if predictor.gpu else 0
                    )
                )
                for predictor in config.enabled_predictors
            ],
            'oksp-predict-generate-model'
        )
    elif args.cmd == 'predict-import':
        print(f'{datetime.now()} Loading data...')
        if args.file.endswith('.csv'):
            data = pd.read_csv(args.file)
        elif args.file.endswith('.tsv'):
            data = pd.read_csv(args.file, sep='\t')
        elif args.file.endswith('.pkl'):
            data = pd.read_pickle(args.file)
        else:
            parser.error('Unsupported file extension for file', args.file)

        print(f'{datetime.now()} Importing...')
        predictor_names = list(itertools.chain.from_iterable(
            predictor.prediction_names for predictor in config.enabled_predictors if not predictor.uses_experimental_reactivities
        ))
        available_columns = list(colname for colname in data.columns if colname in predictor_names)
        with PredictionDB(pred_db_path, 'c') as preddb:
            preddb.upsert_success(
                (
                    (sequence, structure, None)
                    for col in set(predictor_names).intersection(available_columns)
                    for (sequence, structure) in data[['sequence', col]].itertuples(False)
                ),
                args.override
            )
        
        print(f'{datetime.now()} Completed')
    else:
        print(f'{datetime.now()} Loading data...')
        source_data: pd.DataFrame = config.filter_for_computation(load_sources(config.source_files))

        if args.cmd == 'predict-forecast' or args.cmd == 'predict':
            print(f'{datetime.now()} Generating tasks...')
            pred_tasks = []
            with PredictionDB(pred_db_path, 'c') as preddb:
                for predictor in config.enabled_predictors:
                    if not predictor.uses_experimental_reactivities:
                        for sequence in source_data['sequence'].unique():
                            if all(
                                preddb.curr_status(name, sequence, None) != PredictionStatus.SUCCESS
                                for name in predictor.prediction_names
                            ): continue
                            resources = predictor.approximate_resources(sequence)
                            resources.max_runtime *= config.runtime_buffer
                            resources.memory *= config.memory_buffer
                            resources.gpu_memory *= config.gpu_memory_buffer
                            pred_tasks.append(
                                Task(
                                    Runnable.create(predict)(predictor, sequence, None, pred_db_path),
                                    
                                )
                            )
                    else:
                        for sequence, reactivity in source_data[['sequence', 'reactivity']].itertuples(False):
                            if all(
                                preddb.curr_status(name, sequence, reactivity) != PredictionStatus.SUCCESS
                                for name in predictor.prediction_names
                            ): continue
                            resources = predictor.approximate_resources(sequence)
                            resources.max_runtime *= config.runtime_buffer
                            resources.memory *= config.memory_buffer
                            resources.gpu_memory *= config.gpu_memory_buffer
                            pred_tasks.append(
                                Task(
                                    Runnable.create(predict)(predictor, sequence, reactivity, pred_db_path),
                                    resources
                                )
                            )

            if args.cmd == 'predict-forecast':
                print(f'{datetime.now()} Starting forecast...')
                config.runner.forecast(pred_tasks)
            else:
                print(f'{datetime.now()} Starting run...')
                with PredictionDB(pred_db_path, 'c') as preddb:
                    def on_queued(tasks: Iterable[Task]):
                        preddb.upsert_queued(
                            (task.runnable.args[0], task.runnable.args[1], task.runnable.args[2]) for task in tasks
                        )
                    config.runner.run(pred_tasks, 'oksp-predict', on_queued)
                print(f'{datetime.now()} Completed')
    
if __name__ == '__main__':
    run_cli()
