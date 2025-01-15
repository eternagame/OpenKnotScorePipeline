import traceback
from dataclasses import dataclass
from enum import Enum
import os
from datetime import timedelta
from pathlib import Path
import sqlite3
import pickle
from xxhash import xxh3_64_digest
from ...substation.deferred import Deferred, register_deferred
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

        self._cx = sqlite3.connect(f"{pathobj.absolute().as_uri()}?mode={cxflag}", uri=True)

        self._cx.executescript(
            f'''
            CREATE TABLE IF NOT EXISTS predictors (
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sequences (
                sequence TEXT NOT NULL,
                hash BLOB UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reactivities (
                reactivities BLOB NOT NULL,
                hash BLOB UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS status (
                status TEXT UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS structures (
                structure TEXT NOT NULL,
                hash BLOB UNIQUE NOT NULL
            );
            CREATE TABLE IF NOT EXISTS errors (
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
        self._cx.executemany('INSERT INTO status VALUES (?) ON CONFLICT DO NOTHING', [(PredictionStatus.SUCCESS.value,), (PredictionStatus.FAILED.value,), (PredictionStatus.QUEUED.value,)])

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.commit()
        self.close()

    def commit(self):
        self._cx.commit()

    def close(self):
        self._cx.close()
    
    def set_queued(self, predictor: str, sequence: str, reactivities: list[float]):
        reactivities_pkl = pickle.dumps(reactivities)
        sequence_hash = xxh3_64_digest(sequence)
        reactivities_hash = xxh3_64_digest(reactivities_pkl)
        self._cx.execute('INSERT INTO predictors VALUES(?) ON CONFLICT DO NOTHING', (predictor,)).close()
        self._cx.execute('INSERT INTO sequences VALUES(?, ?) ON CONFLICT DO NOTHING', (sequence, sequence_hash)).close()
        self._cx.execute('INSERT INTO reactivities VALUES(?, ?) ON CONFLICT DO NOTHING', (reactivities_pkl, reactivities_hash)).close()
        self._cx.execute(
            '''
            INSERT INTO predictions(predictor, sequence, reactivities, status) VALUES(
                (SELECT ROWID FROM predictors WHERE name=?),
                (SELECT ROWID FROM sequences WHERE hash=?),
                (SELECT ROWID FROM reactivities WHERE hash=?),
                (SELECT ROWID FROM status where status=?)
            ) ON CONFLICT DO UPDATE SET status=(SELECT ROWID FROM status where status=?)
            ''',
            (predictor, sequence_hash, reactivities_hash, PredictionStatus.QUEUED.value, PredictionStatus.QUEUED.value),
        ).close()
        

    def set_success(self, predictor: str, sequence: str, reactivities: list[float], structure: str):
        structure_hash = xxh3_64_digest(structure)
        self._cx.execute('INSERT INTO structures VALUES(?, ?) ON CONFLICT DO NOTHING', (structure, structure_hash)).close()
        self._cx.execute(
            '''
            UPDATE predictions SET status=(SELECT ROWID FROM status where status=?), result=(SELECT ROWID FROM structures WHERE hash=?)
            WHERE
                predictor=(SELECT ROWID FROM predictors WHERE name=?)
                AND sequence=(SELECT ROWID FROM sequences WHERE hash=?)
                AND reactivities=(SELECT ROWID FROM reactivities WHERE hash=?)
            ''',
            (PredictionStatus.SUCCESS.value, structure_hash, predictor, xxh3_64_digest(sequence), xxh3_64_digest(pickle.dumps(reactivities)))
        ).close()

    def set_failure(self, predictor: str, sequence: str, reactivities: list[float], error: str):
        error_hash = xxh3_64_digest(error)
        self._cx.execute('INSERT INTO errors VALUES(?, ?) ON CONFLICT DO NOTHING', (error, error_hash)).close()
        self._cx.execute(
            '''
            UPDATE predictions SET status=(SELECT ROWID FROM status where status=?), result=(SELECT ROWID FROM errors WHERE hash=?)
            WHERE
                predictor=(SELECT ROWID FROM predictors WHERE name=?)
                AND sequence=(SELECT ROWID FROM sequences WHERE hash=?)
                AND reactivities=(SELECT ROWID FROM reactivities WHERE hash=?)
            ''',
            (PredictionStatus.FAILED.value, error_hash, predictor, xxh3_64_digest(sequence), xxh3_64_digest(pickle.dumps(reactivities)))
        ).close()

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
                db.set_success(result.predictor, result.seq, result.reactivities, result.output)
            else:
                db.set_failure(result.predictor, result.seq, result.reactivities, result.output)

deferred = dict[str, Deferred]()

def predict(predictor: Predictor, seq: str, reactivities: list[float], db_path: str):
    try:
        predictions = predictor.run(seq, reactivities)
        for (name, pred) in predictions.items():
            failed = all(char == 'x' for char in pred)
            if failed:
                print(f'FAILED: predictor={name}, seq={seq}, reactivities={reactivities}')
                print('Arnie returned all "x"')
            results.append(Result(
                name,
                seq,
                reactivities,
                not failed,
                pred if not failed else 'Arnie returned all "x"'
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
