'''
Custom file format for storing pipeline data, optimized for random access (without fully
loading into memory) and data deduplication

This takes inspiration from dbm.sqite, sqlitedict, and hdf5 in various forms
'''

from typing import Any
from contextlib import closing
import os
import re
import pickle
import sqlite3
from pathlib import Path
from xxhash import xxh3_64_digest

class DB:
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
        self._cx.execute(
            f'''
            CREATE TABLE IF NOT EXISTS collections (
                name TEXT UNIQUE NOT NULL
            )'''
        ).close()
        self._collections = dict[str, int]()
        with closing(self._cx.execute('SELECT name, ROWID FROM collections')) as cur:
            for (name, rowid) in cur:
                self._collections[name] = rowid

        self._cx.execute(
            f'''
            CREATE TABLE IF NOT EXISTS entries (
                collection INTEGER NOT NULL,
                key BLOB NOT NULL,
                hash BLOB NOT NULL
            )'''
        ).close()

        self._cx.execute(
            f'''
            CREATE TABLE IF NOT EXISTS data (
                hash BLOB UNIQUE NOT NULL,
                value BLOB NOT NULL
            )'''
        ).close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.commit()
        self.close()

    def commit(self):
        self._cx.commit()

    def close(self):
        self._cx.close()

    def _create_collection(self, name: str):
        if not re.match(r'^[a-zA-Z]\w*$', name):
            raise ValueError(f'Invalid collection name: {name}')
        
        with closing(self._cx.execute(
            'INSERT INTO collections(name) VALUES(?) ON CONFLICT DO UPDATE SET name=name RETURNING ROWID',
            (name,)
        )) as cur:
            (rowid,) = cur.fetchone()
        self._collections[name] = rowid

    @staticmethod
    def _encode_key(key: bytes | str | int):
        if isinstance(key, str):
            return b's' + key.encode()
        elif isinstance(key, int):
            if key < 0:
                raise ValueError('Negative integers as keys is not supported')
            return b'i' + key.to_bytes((key.bit_length() + 7) // 8)
        else:
            return b'b' + key

    def insert(self, collection: str, value: Any, key: str | bytes | int = None):
        if not collection in self._collections:
            self._create_collection(collection)

        pkl = pickle.dumps(value)
        hash = xxh3_64_digest(pkl)
        key = key if key else hash

        with closing(self._cx.execute('SELECT 1 FROM data WHERE hash=?', (hash,))) as cur:
            exists = cur.fetchone() is not None
        if not exists:
            self._cx.execute(
                'INSERT INTO data(hash, value) VALUES(?, ?)',
                (hash, pkl)
            ).close()
        self._cx.execute(
            'INSERT INTO entries(collection, key, hash) VALUES(?, ?, ?)',
            (self._collections[collection], self._encode_key(key), hash)
        ).close()

        return key

    def get(self, collection: str, key: str | bytes | int):
        with closing(self._cx.execute(
            'SELECT data.value FROM entries LEFT JOIN data on entries.hash=data.hash WHERE entries.collection=? AND entries.key=?',
            (self._collections[collection], self._encode_key(key))
        )) as cur:
            res = cur.fetchone()
        if not res:
            raise KeyError(f'Could not find item. collection={collection} key={key}')
        value = res[0]
        return pickle.loads(value)
