from typing import Iterable
from contextlib import closing
import traceback
from dataclasses import dataclass
from enum import Enum
import os
from datetime import timedelta
from pathlib import Path
import sqlite3
import pickle
from xxhash import xxh3_64_digest
from arnie.utils import convert_dotbracket_to_bp_list, convert_bp_list_to_dotbracket
from ...substation.deferred import Deferred, register_deferred
from ...substation.scheduler.domain import Task
from .predictors import Predictor

class PredictionStatus(Enum):
    QUEUED='queued'
    SUCCESS='success'
    FAILED='failed'

class PredictionDB:
    def __init__(self, path: str, flag="r", mode=0o666):
        pathobj = Path(os.fsdecode(path))

        if flag == "r":
            cxflag = "ro"
        elif flag == "w":
            cxflag = "rw"
        elif flag == "c":
            cxflag = "rwc"
            pathobj.touch(mode=mode, exist_ok=True)
        elif flag == "n":
            cxflag = "rwc"
            pathobj.unlink(missing_ok=True)
            pathobj.touch(mode=mode)
        else:
            raise ValueError(f"Flag must be one of 'r', 'w', 'c', or 'n', not {flag!r}")

        self._cx = sqlite3.connect(f"{pathobj.absolute().as_uri()}?mode={cxflag}", uri=True, timeout=60)

        self._cx.executescript(
            f'''
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictors (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sequences (
                id INTEGER PRIMARY KEY,
                sequence TEXT NOT NULL,
                hash BLOB UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reactivities (
                id INTEGER PRIMARY KEY,
                reactivities BLOB NOT NULL,
                hash BLOB UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS status (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS structures (
                id INTEGER PRIMARY KEY,
                structure TEXT NOT NULL,
                hash BLOB UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY,
                error TEXT NOT NULL,
                hash BLOB UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                predictor INTEGER NOT NULL,
                sequence INTEGER NOT NULL,
                reactivities INTEGER NOT NULL,
                status INTEGER NOT NULL,
                result INTEGER
            );
            CREATE UNIQUE INDEX IF NOT EXISTS prediction_args ON predictions(predictor, sequence, reactivities); 
            '''
        )
        self._cx.execute("INSERT INTO meta VALUES('version', '1') ON CONFLICT DO UPDATE SET value='1'")
        self._cx.executemany('INSERT INTO status(name) VALUES (?) ON CONFLICT DO NOTHING', [(PredictionStatus.SUCCESS.value,), (PredictionStatus.FAILED.value,), (PredictionStatus.QUEUED.value,)])

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.commit()
        self.close()

    def commit(self):
        self._cx.commit()

    def close(self):
        self._cx.close()
    
    def curr_status(self, predictor: str, sequence: str, reactivities: list[float]):
        with closing(self._cx.execute(
            '''
            SELECT status.name FROM predictions LEFT JOIN status on status.id=predictions.status
            WHERE
                predictor=(SELECT id FROM predictors WHERE name=?)
                AND sequence=(SELECT id FROM sequences WHERE hash=?)
                AND reactivities=(SELECT id FROM reactivities WHERE hash=?)
            ''',
            (predictor, xxh3_64_digest(sequence), xxh3_64_digest(pickle.dumps(reactivities)))
        )) as cur:
            val = cur.fetchone()
            if val:
                return PredictionStatus(val[0])
            return None

    def upsert_queued(self, inputs: Iterable[tuple[Predictor, str, list[float]]]):
        hash_cache = dict[int, bytes]()
        name_cache = set()

        for (predictor, sequence, reactivities) in inputs:
            for name in predictor.prediction_names:
                if not name in name_cache:
                    self._cx.execute('INSERT INTO predictors(name) VALUES(?) ON CONFLICT DO NOTHING', (name,)).close()
                    name_cache.add(name)
                sequence_hash = hash_cache.get(id(sequence))
                if not sequence_hash:
                    sequence_hash = xxh3_64_digest(sequence)
                    hash_cache[id(sequence)] = sequence_hash
                    self._cx.execute('INSERT INTO sequences(sequence, hash) VALUES(?, ?) ON CONFLICT DO NOTHING', (sequence, sequence_hash)).close()
                reactivities_hash = hash_cache.get(id(reactivities))
                if not reactivities_hash:
                    reactivities_pkl = pickle.dumps(reactivities)
                    reactivities_hash = xxh3_64_digest(reactivities_pkl)
                    hash_cache[id(reactivities)] = reactivities_hash
                    self._cx.execute('INSERT INTO reactivities(reactivities, hash) VALUES(?, ?) ON CONFLICT DO NOTHING', (reactivities_pkl, reactivities_hash)).close()
                
                self._cx.execute(
                    '''
                    INSERT INTO predictions(predictor, sequence, reactivities, status) VALUES(
                        (SELECT id FROM predictors WHERE name=?),
                        (SELECT id FROM sequences WHERE hash=?),
                        (SELECT id FROM reactivities WHERE hash=?),
                        (SELECT id FROM status where name=?)
                    ) ON CONFLICT DO UPDATE SET status=(SELECT id FROM status where name=?)
                    ''',
                    (name, sequence_hash, reactivities_hash, PredictionStatus.QUEUED.value, PredictionStatus.QUEUED.value),
                ).close()

    def upsert_success(self, inputs: Iterable[tuple[Predictor, str, list[float], str]], override: bool):
        hash_cache = dict[int, bytes]()
        name_cache = set()

        for (predictor, sequence, reactivities, structure) in inputs:
            for name in predictor.prediction_names:
                if not name in name_cache:
                    self._cx.execute('INSERT INTO predictors(name) VALUES(?) ON CONFLICT DO NOTHING', (name,)).close()
                    name_cache.add(name)
                sequence_hash = hash_cache.get(id(sequence))
                if not sequence_hash:
                    sequence_hash = xxh3_64_digest(sequence)
                    hash_cache[id(sequence)] = sequence_hash
                    self._cx.execute('INSERT INTO sequences(sequence, hash) VALUES(?, ?) ON CONFLICT DO NOTHING', (sequence, sequence_hash)).close()
                reactivities_hash = hash_cache.get(id(reactivities))
                if not reactivities_hash:
                    reactivities_pkl = pickle.dumps(reactivities)
                    reactivities_hash = xxh3_64_digest(reactivities_pkl)
                    hash_cache[id(reactivities)] = reactivities_hash
                    self._cx.execute('INSERT INTO reactivities(reactivities, hash) VALUES(?, ?) ON CONFLICT DO NOTHING', (reactivities_pkl, reactivities_hash)).close()
                structure_hash = hash_cache.get(id(structure))
                if not structure_hash:
                    structure_hash = xxh3_64_digest(structure)
                    hash_cache[id(structure)] = structure_hash
                    self._cx.execute('INSERT INTO structures(structure, hash) VALUES(?, ?) ON CONFLICT DO NOTHING', (structure, structure_hash)).close()
                
                if override:
                    self._cx.execute(
                        f'''
                        INSERT INTO predictions(predictor, sequence, reactivities, result, status) VALUES(
                            (SELECT id FROM predictors WHERE name=?),
                            (SELECT id FROM sequences WHERE hash=?),
                            (SELECT id FROM reactivities WHERE hash=?),
                            (SELECT id FROM structures WHERE hash=?),
                            (SELECT id FROM status where name=?)
                        ) ON CONFLICT DO UPDATE SET status=(SELECT id FROM status where name=?), result=(SELECT id FROM structures WHERE hash=?)
                        ''',
                        (name, sequence_hash, reactivities_hash, structure_hash, PredictionStatus.SUCCESS.value, PredictionStatus.SUCCESS.value),
                    ).close()
                else:
                    self._cx.execute(
                        f'''
                        INSERT INTO predictions(predictor, sequence, reactivities, result, status) VALUES(
                            (SELECT id FROM predictors WHERE name=?),
                            (SELECT id FROM sequences WHERE hash=?),
                            (SELECT id FROM reactivities WHERE hash=?),
                            (SELECT id FROM structures WHERE hash=?),
                            (SELECT id FROM status where name=?)
                        ) ON CONFLICT DO UPDATE SET status=(SELECT id FROM status where name=?), result=(SELECT id FROM structures WHERE hash=?) WHERE status=(SELECT id FROM status where name=?)
                        ''',
                        (name, sequence_hash, reactivities_hash, structure_hash, PredictionStatus.SUCCESS.value, PredictionStatus.SUCCESS.value, PredictionStatus.QUEUED.value),
                    ).close()

    def update_success(self, predictor: str, sequence: str, reactivities: list[float], structure: str):
        structure_hash = xxh3_64_digest(structure)
        self._cx.execute('INSERT INTO structures(structure, hash) VALUES(?, ?) ON CONFLICT DO NOTHING', (structure, structure_hash)).close()
        self._cx.execute(
            '''
            UPDATE predictions SET status=(SELECT id FROM status where name=?), result=(SELECT id FROM structures WHERE hash=?)
            WHERE
                predictor=(SELECT id FROM predictors WHERE name=?)
                AND sequence=(SELECT id FROM sequences WHERE hash=?)
                AND reactivities=(SELECT id FROM reactivities WHERE hash=?)
            ''',
            (PredictionStatus.SUCCESS.value, structure_hash, predictor, xxh3_64_digest(sequence), xxh3_64_digest(pickle.dumps(reactivities)))
        ).close()

    def update_failure(self, predictor: str, sequence: str, reactivities: list[float], error: str):
        error_hash = xxh3_64_digest(error)
        self._cx.execute('INSERT INTO errors(error, hash) VALUES(?, ?) ON CONFLICT DO NOTHING', (error, error_hash)).close()
        self._cx.execute(
            '''
            UPDATE predictions SET status=(SELECT id FROM status where name=?), result=(SELECT id FROM errors WHERE hash=?)
            WHERE
                predictor=(SELECT id FROM predictors WHERE name=?)
                AND sequence=(SELECT id FROM sequences WHERE hash=?)
                AND reactivities=(SELECT id FROM reactivities WHERE hash=?)
            ''',
            (PredictionStatus.FAILED.value, error_hash, predictor, xxh3_64_digest(sequence), xxh3_64_digest(pickle.dumps(reactivities)))
        ).close()

    def get_predictions(self, sequence: str, reactivities: list[float], nonreactive_predictors: list[str], reactive_predictors: list[str]):
        with closing(self._cx.execute(
            f'''
            SELECT predictors.name, structures.structure
            FROM
                predictions
                LEFT JOIN structures on structures.id=predictions.result
                LEFT JOIN predictors on predictors.id=predictions.predictor
            WHERE
                status=(SELECT id FROM status WHERE name=?)
                AND sequence=(SELECT id FROM sequences WHERE hash=?)
                AND (
                    (
                        predictor in (SELECT id FROM predictors WHERE name in ({','.join('?'*len(reactive_predictors))}))
                        AND reactivities=(SELECT id FROM reactivities WHERE hash=?)
                    ) OR predictor in (SELECT id FROM predictors WHERE name in ({','.join('?'*len(nonreactive_predictors))}))
                )
            ''',
            (PredictionStatus.SUCCESS.value, xxh3_64_digest(sequence), *reactive_predictors, xxh3_64_digest(pickle.dumps(reactivities)), *nonreactive_predictors)
        )) as cur:
            return {predictor: structure for (predictor, structure) in cur.fetchall()}

    def get_failed(self, sequence: str, reactivities: list[float], nonreactive_predictors: list[str], reactive_predictors: list[str]):
        with closing(self._cx.execute(
            f'''
            SELECT predictors.name, errors.error
            FROM
                predictions
                LEFT JOIN errors on errors.id=predictions.result
                LEFT JOIN predictors on predictors.id=predictions.predictor
            WHERE
                status=(SELECT id FROM status WHERE name=?)
                AND sequence=(SELECT id FROM sequences WHERE hash=?)
                AND (
                    (
                        predictor in (SELECT id FROM predictors WHERE name in ({','.join('?'*len(reactive_predictors))}))
                        AND reactivities=(SELECT id FROM reactivities WHERE hash=?)
                    ) OR predictor in (SELECT id FROM predictors WHERE name in ({','.join('?'*len(nonreactive_predictors))}))
                )
            ''',
            (PredictionStatus.FAILED.value, xxh3_64_digest(sequence), *reactive_predictors, xxh3_64_digest(pickle.dumps(reactivities)), *nonreactive_predictors)
        )) as cur:
            return {predictor: structure for (predictor, structure) in cur.fetchall()}

    def get_prediction_success(self, sequence: str, reactivities: list[float], nonreactive_predictors: list[str], reactive_predictors: list[str]):
        with closing(self._cx.execute(
            f'''
            SELECT predictors.name
            FROM
                predictions
                LEFT JOIN errors on errors.id=predictions.result
                LEFT JOIN predictors on predictors.id=predictions.predictor
            WHERE
                status=(SELECT id FROM status WHERE name=?)
                AND sequence=(SELECT id FROM sequences WHERE hash=?)
                AND (
                    (
                        predictor in (SELECT id FROM predictors WHERE name in ({','.join('?'*len(reactive_predictors))}))
                        AND reactivities=(SELECT id FROM reactivities WHERE hash=?)
                    ) OR predictor in (SELECT id FROM predictors WHERE name in ({','.join('?'*len(nonreactive_predictors))}))
                )
            ''',
            (PredictionStatus.SUCCESS.value, xxh3_64_digest(sequence), *reactive_predictors, xxh3_64_digest(pickle.dumps(reactivities)), *nonreactive_predictors)
        )) as cur:
            return {predictor: True for (predictor,) in cur.fetchall()}

    def clear_predictor(self, predictor: str):
        self._cx.execute('DELETE FROM predictions WHERE predictor=(SELECT id FROM predictors WHERE name=?)', (predictor,)).close()

@dataclass
class Result:
    predictor: str
    seq: str
    reactivities: list[float]
    success: bool
    output: str

results = list[Result]()

def flush_predictions(db_path: str):
    with PredictionDB(db_path, 'w') as db:
        while results:
            result = results.pop()
            if result.success:
                db.update_success(result.predictor, result.seq, result.reactivities, result.output)
            else:
                db.update_failure(result.predictor, result.seq, result.reactivities, result.output)

deferred = dict[str, Deferred]()

def predict(predictor: Predictor, seq: str, reactivities: list[float], db_path: str):
    try:
        predictions = predictor.run(seq, reactivities)
        for (name, pred) in predictions.items():
            result = pred
            failed = False

            if all(char == 'x' for char in pred):
                failed = True
                print(f'FAILED: predictor={name}, seq={seq}, reactivities={reactivities}')
                print('Arnie returned all "x"')
                result = 'Arnie returned all "x"'
            else:
                try:
                    convert_bp_list_to_dotbracket(convert_dotbracket_to_bp_list(pred, allow_pseudoknots=True), len(pred))
                except Exception:
                    failed = True
                    err = traceback.format_exc()
                    print(f'FAILED: predictor={name}, seq={seq}, reactivities={reactivities}')
                    print('Invalid dot bracket')
                    print(err)
                    result = 'Invalid dot bracket\n' + err

            results.append(Result(
                name,
                seq,
                reactivities,
                not failed,
                result
            ))
    except Exception:
        err = traceback.format_exc()
        print(f'FAILED: predictors={predictor.prediction_names}, seq={seq}, reactivities={reactivities}')
        print(err)
        for name in predictor.prediction_names:
            results.append(Result(
                name,
                seq,
                reactivities,
                False,
                err
            ))

    if not deferred.get(db_path):
        deferred[db_path] = Deferred(
            lambda: flush_predictions(db_path),
            # Save if it's been at least this long since our last save, to prevent losing too much
            # work if we crash or are killed
            flush_after_time=timedelta(minutes=5),
            # If we have accumulated this many results since our last save, go ahead and flush them
            # even if we haven't hit the time yet to avoid consuming excess memory
            flush_after_count=2500
        )
    register_deferred(deferred[db_path])
