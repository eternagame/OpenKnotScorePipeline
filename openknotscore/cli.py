import argparse
import importlib
import os
from os import path
from datetime import datetime
import itertools
import math
from tqdm import tqdm
import pandas as pd
from typing import Iterable
from .pipeline.import_source import load_sources, load_extension_sources
from .pipeline.prediction.sample import generate_predictor_resource_model, MODEL_TIMEOUT
from .pipeline.prediction.predict import predict, PredictionDB, PredictionStatus
from .pipeline import scoring
from .substation.scheduler.domain import Task, Runnable, UtilizedResources
from .config import OKSPConfig

def run_cli():
    # NOTE: Any configuration needed to ensure reproducibility should be provided
    # via the config argument/class, NOT command line flags!
    parser = argparse.ArgumentParser()

    parser.add_argument(dest='module', help='python module path containing Config class')

    subparsers = parser.add_subparsers(dest='cmd', help='subcommand', required=True)
    predict_parser = subparsers.add_parser('predict', help='generate missing structure predictions')
    predict_parser.add_argument('--skip-failed', action='store_true', dest='skip_failed', help='Dont rerun failed predictions')
    subparsers.add_parser('score', help='compute scores using previously generated predictions and export to the final output files')
    forecast_parser = subparsers.add_parser('predict-forecast', help='compute the total number of core-hours required for computation and minimum per-job requirements')
    forecast_parser.add_argument('--skip-failed', action='store_true', dest='skip_failed', help='Dont rerun failed predictions')
    model_parser = subparsers.add_parser('predict-generate-model', help='run predictors with sample inputs and generate resource usage models')
    model_parser.add_argument('--cpus', type=int, dest='cpus', help='cpus to allocate per predictor, in MB (limits not supported for local runner)', default=1)
    model_parser.add_argument('--max-memory', type=int, dest='max_memory', help='maximum memory to allocate per predictor, in MB (limits not supported for local runner)', default=1000*16)
    model_parser.add_argument('--max-gpu-memory', type=int, dest='max_gpu_memory', help='maximum GPU memory to allocate per predictor if it supports GPU, in MB (limits not supported for local runner)', default=1000*70)
    import_parser = subparsers.add_parser('predict-import', help='import pre-computed structures from another file into the pipeline database')
    import_parser.add_argument(dest='file', help='path to the file to import structures from')
    import_parser.add_argument('--override', action='store_true', dest='override', help='override values in the database if they already exist')
    check_failed_parser = subparsers.add_parser('predict-check-failed', help='show information about missing or failed predictions')
    check_failed_parser.add_argument('--show-errors', action='store_true', dest='show_errors', help='if errors were encountered, print what they were', default=False)
    check_failed_parser.add_argument('--nrows', type=int, dest='nrows', help='for missing/failed predictions, print the first n rows of source data which failed per unique failure', default=0)
    clear_parser = subparsers.add_parser('predict-clear', help='clear predictions for a given predictor')
    clear_parser.add_argument('predictor')

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
                        math.ceil(MODEL_TIMEOUT * config.runtime_buffer), MODEL_TIMEOUT, MODEL_TIMEOUT,
                        args.cpus,
                        # Note we don't use the runtime/memory buffer here, because the user is specifying
                        # an actual cap they want, we're not adding overage for an estimate
                        args.max_memory * 1024 * 1024,
                        args.max_gpu_memory * 1024 * 1024 if predictor.gpu else 0
                    )
                )
                for predictor in config.enabled_predictors
            ],
            'oksp-predict-generate-model'
        )
    elif args.cmd == 'predict-import':
        print(f'{datetime.now()} Loading data...')
        def parse_list(val: str):
            return [None if x.strip() in ('None', 'nan', 'NaN') else float(x) for x in val.removeprefix('[').removesuffix(']').split(',')]
        if args.file.endswith('.csv'):
            data = pd.read_csv(args.file, converters={'reactivity': parse_list})
        elif args.file.endswith('.tsv'):
            data = pd.read_csv(args.file, sep='\t', converters={'reactivity': parse_list})
        elif args.file.endswith('.pkl'):
            data = pd.read_pickle(args.file)
        else:
            parser.error('Unsupported file extension for file', args.file)

        print(f'{datetime.now()} Importing...')
        predictor_names = list(itertools.chain.from_iterable(
            predictor.prediction_names for predictor in config.enabled_predictors if not predictor.uses_experimental_reactivities
        ))
        available_columns = list(
            colname for colname in data.columns
            if config.imported_column_map.get(colname, colname.removesuffix('_PRED')) in predictor_names
        )

        def is_valid_structure(structure, sequence):
            if pd.isna(structure): return False
            if all([char == "x" for char in structure]): return False
            if structure.strip() == '': return False
            if len(structure) != len(sequence): return False

            try:
                convert_bp_list_to_dotbracket(convert_dotbracket_to_bp_list(structure, allow_pseudoknots=True), len(structure))
            except:
                return False

            return True

        with PredictionDB(pred_db_path, 'c') as preddb:
            preddb.upsert_success(
                (
                    (config.imported_column_map.get(colname, colname.removesuffix('_PRED')), sequence, None, structure)
                    for colname in available_columns
                    for (idx, sequence, structure) in data[['sequence', colname]].itertuples(True)
                    if is_valid_structure(structure, sequence)
                ),
                args.override
            )
        if 'reactivity' in data.columns:
            reactive_predictor_names = list(itertools.chain.from_iterable(
                predictor.prediction_names for predictor in config.enabled_predictors if predictor.uses_experimental_reactivities
            ))
            reactive_available_columns = list(
                colname for colname in data.columns
                if config.imported_column_map.get(colname, colname.removesuffix('_PRED')) in reactive_predictor_names
            )
            with PredictionDB(pred_db_path, 'c') as preddb:
                preddb.upsert_success(
                    (
                        (config.imported_column_map.get(colname, colname.removesuffix('_PRED')), sequence, reactivity, structure)
                        for colname in reactive_available_columns
                        for (sequence, reactivity, structure) in data[['sequence', 'reactivity', colname]].itertuples(False)
                        if is_valid_structure(structure, sequence)
                    ),
                    args.override
                )
        
        print(f'{datetime.now()} Completed')
    elif args.cmd == 'predict-clear':
        print('Clearing...')
        with PredictionDB(pred_db_path, 'w') as preddb:
            preddb.clear_predictor(args.predictor)
    else:
        print(f'{datetime.now()} Loading data...')
        data: pd.DataFrame = config.preprocess_data(
            load_extension_sources(
                config.extension_source_files,
                load_sources(config.source_files)
            )
        )

        if args.cmd == 'predict-forecast' or args.cmd == 'predict':
            print(f'{datetime.now()} Generating tasks...')
            pred_tasks = []
            with PredictionDB(pred_db_path, 'c') as preddb:
                for predictor in config.enabled_predictors:
                    if not predictor.uses_experimental_reactivities:
                        for sequence in data['sequence'].unique():
                            if all(
                                preddb.curr_status(name, sequence, None) in (
                                    [PredictionStatus.SUCCESS, PredictionStatus.FAILED] if args.skip_failed else [PredictionStatus.SUCCESS]
                                )
                                for name in predictor.prediction_names
                            ): continue
                            resources = predictor.approximate_resources(sequence)
                            resources.max_runtime = math.ceil(resources.max_runtime * config.runtime_buffer)
                            resources.memory = math.ceil(resources.memory * config.memory_buffer)
                            resources.gpu_memory = math.ceil(resources.gpu_memory * config.gpu_memory_buffer)
                            pred_tasks.append(
                                Task(
                                    Runnable.create(predict)(predictor, sequence, None, pred_db_path),
                                    resources
                                )
                            )
                    else:
                        for sequence, reactivity in data[['sequence', 'reactivity']].itertuples(False):
                            if all(
                                preddb.curr_status(name, sequence, reactivity) == PredictionStatus.SUCCESS
                                for name in predictor.prediction_names
                            ): continue
                            resources = predictor.approximate_resources(sequence)
                            resources.max_runtime = math.ceil(resources.max_runtime * config.runtime_buffer)
                            resources.memory = math.ceil(resources.memory * config.memory_buffer)
                            resources.gpu_memory = math.ceil(resources.gpu_memory * config.gpu_memory_buffer)
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
                def on_queued(tasks: Iterable[Task]):
                    with PredictionDB(pred_db_path, 'c') as preddb:
                        preddb.upsert_queued(
                            (task.runnable.args[0], task.runnable.args[1], task.runnable.args[2]) for task in tasks
                        )
                config.runner.run(pred_tasks, 'oksp-predict', on_queued)
                print(f'{datetime.now()} Completed')
    
        if args.cmd == 'predict-check-failed':
            nonreactive_prediction_names = list(itertools.chain.from_iterable(
                predictor.prediction_names for predictor in config.enabled_predictors
                if not predictor.uses_experimental_reactivities
            ))
            reactive_prediction_names = list(itertools.chain.from_iterable(
                predictor.prediction_names for predictor in config.enabled_predictors
                if predictor.uses_experimental_reactivities
            ))
            prediction_names = [*nonreactive_prediction_names, *reactive_prediction_names]
            with PredictionDB(pred_db_path, 'c') as preddb:
                tqdm.pandas(desc='Retrieving errors')
                errors = data.progress_apply(
                    lambda row: preddb.get_failed(row['sequence'], row.get('reactivity'), nonreactive_prediction_names, reactive_prediction_names),
                    axis=1,
                    result_type='expand'
                )
                data = pd.merge(data,errors,how="left",left_index=True,right_index=True)
                tqdm.pandas(desc='Retrieving predictions')
                exists = data.progress_apply(
                    lambda row: preddb.get_prediction_success(row['sequence'], row.get('reactivity'), nonreactive_prediction_names, reactive_prediction_names),
                    axis=1,
                    result_type='expand'
                )
                data = data.combine_first(exists)

            for col in errors.columns:
                print(f'Predictor {col} had {errors[col].dropna().nunique()} unique errors across {errors[col].dropna().count()} solutions')
                if args.show_errors:
                    print('==============================')
                    for error in errors[col].dropna().unique():
                        print(error)
                        print('==============================')
                        
                        if args.nrows > 0:
                            print(data[[col for col in data.columns if col not in prediction_names]][data[col] == error][:args.nrows].to_string())
            for col in prediction_names:
                if not col in data.columns:
                    print(f'Predictor {col} is not present for any solutions')
                else:
                    missing = data[col].isna().sum()
                    if missing > 0:
                        print(f'Predictor {col} is missing for {data[col].isna().sum()} solutions')
                        if args.nrows > 0:
                            print(data[[col for col in data.columns if col not in prediction_names]][data[col] == error][:args.nrows].to_string())
            

        if args.cmd == 'score':
            nonreactive_prediction_names = list(itertools.chain.from_iterable(
                predictor.prediction_names for predictor in config.enabled_predictors
                if not predictor.uses_experimental_reactivities
            ))
            reactive_prediction_names = list(itertools.chain.from_iterable(
                predictor.prediction_names for predictor in config.enabled_predictors
                if predictor.uses_experimental_reactivities
            ))
            tqdm.pandas(desc='Retrieving predictions')
            with PredictionDB(pred_db_path, 'c') as preddb:
                preds = data.progress_apply(
                    lambda row: preddb.get_predictions(row['sequence'], row.get('reactivity'), nonreactive_prediction_names, reactive_prediction_names),
                    axis=1,
                    result_type='expand'
                )
                data = pd.merge(data,preds,how="left",left_index=True,right_index=True)

            prediction_names = [
                name for name in [*nonreactive_prediction_names, *reactive_prediction_names, 'target_structure'] if name in data.columns
            ]

            def getECSperRow(row):
                df = row[prediction_names].apply(
                    scoring.calculateEternaClassicScore,
                    args=(row['reactivity'], row['score_start_idx'] - 1, row['score_end_idx'] - 1),
                    filter_singlets=config.filter_singlets
                )
                df = df.add_suffix('_ECS')
                return df

            tqdm.pandas(desc="Calculating Eterna Classic Score")
            ecs = data.progress_apply(getECSperRow,axis=1,result_type='expand')
            data = pd.merge(data,ecs,how="left",left_index=True,right_index=True)

            def getCPQperRow(row):
                # Apply the scoring function to each model prediction in the row
                df = row[prediction_names].apply(
                    scoring.calculateCrossedPairQualityScore,
                    args=(row['reactivity'], row['score_start_idx'] - 1, row['score_end_idx'] - 1),
                    filter_singlets=config.filter_singlets
                )
                df = df.add_suffix('_CPQ')
                return df

            tqdm.pandas(desc="Calculating Crossed Pair Quality Score")
            cpq = data.progress_apply(getCPQperRow,axis=1,result_type='expand')
            data = pd.merge(data,cpq,how="left",left_index=True,right_index=True)

            def getOKSperRow(row):
                # Missing or failed structures should not be considered as candidates
                tags = [tag for tag in prediction_names if not pd.isna(row[tag])]
                return scoring.calculateOpenKnotScore(row, tags)

            tqdm.pandas(desc="Calculating OpenKnotScore")
            oks = data.progress_apply(getOKSperRow,axis=1)
            data = pd.merge(data,oks,how="left",left_index=True,right_index=True)

            data = data.rename(columns={c: c+'_PRED' for c in prediction_names})

            print('Writing output files...')
            for output in config.output_configs:
                output.write(data, config)

if __name__ == '__main__':
    run_cli()
