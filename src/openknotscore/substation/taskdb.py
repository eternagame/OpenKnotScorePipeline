from typing import NamedTuple
from pathlib import Path
from contextlib import closing
import itertools
import os
import pickle
import sqlite3
from xxhash import xxh3_64_digest
from .scheduler.domain import Schedule, Runnable

class DBQueue(NamedTuple):
    id: int
    cpus: int
    memory: int
    gpu_id: int | None

class TaskDB:
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

        self._cx.executescript('''
        CREATE TABLE IF NOT EXISTS functions (
            id INTEGER PRIMARY KEY,
            function BLOB NOT NULL,
            hash BLOB UNIQUE NOT NULL
        );
        CREATE TABLE IF NOT EXISTS arguments (
            id INTEGER PRIMARY KEY,
            argument BLOB UNIQUE NOT NULL,
            hash BLOB UNIQUE NOT NULL,
            is_kw INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS queues (
            id INTEGER UNIQUE NOT NULL,
            configuration INTEGER NOT NULL,
            allocation INTEGER NOT NULL,
            cpus INTEGER NOT NULL,
            memory INTEGER NOT NULL,
            gpu_id INTEGER,
            parent INTEGER
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER UNIQUE NOT NULL,
            queue INTEGER NOT NULL,
            function INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS task_args (
            task INTEGER NOT NULL,
            argument INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS task_args_task ON task_args(task);
        ''').close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.commit()
        self.close()

    def commit(self):
        self._cx.commit()

    def close(self):
        self._cx.close()

    def write_schedule(self, schedule: Schedule):
        hash_cache = dict[int, bytes]()

        def functions():
            for task in schedule.tasks:
                if not id(task.runnable.func) in hash_cache:
                    pkl = pickle.dumps(task.runnable.func)
                    hash = xxh3_64_digest(pkl)
                    hash_cache[id(task.runnable.func)] = hash
                    yield (pkl, hash)
        
        def args():
            for task in schedule.tasks:
                for arg in task.runnable.args:
                    if not id(arg) in hash_cache:
                        pkl = pickle.dumps(arg)
                        hash = xxh3_64_digest(pkl)
                        hash_cache[id(arg)] = hash
                        yield (pkl, hash, False)
                for kwarg in task.runnable.kwargs:
                    if not id(kwarg) in hash_cache:
                        pkl = pickle.dumps(kwarg)
                        hash = xxh3_64_digest(pkl)
                        hash_cache[id(kwarg)] = hash
                        yield (pkl, hash, True)

        self._cx.executemany(
            '''
            INSERT INTO functions(function, hash) VALUES(?, ?) ON CONFLICT DO NOTHING
            ''',
            functions()
        ).close()
        self._cx.executemany(
            '''
            INSERT INTO arguments(argument, hash, is_kw) VALUES(?, ?, ?) ON CONFLICT DO NOTHING
            ''',
            args()
        ).close()
        self._cx.executemany(
            '''
            INSERT INTO task_args VALUES(?, (SELECT id FROM arguments WHERE hash=?))
            ''',
            itertools.chain.from_iterable([(task.id, hash_cache[id(arg)]) for arg in task.runnable.args] for task in schedule.tasks)
        ).close()
        self._cx.executemany(
            '''
            INSERT INTO task_args VALUES(?, (SELECT id FROM arguments WHERE hash=?))
            ''',
            itertools.chain.from_iterable([(task.id, hash_cache[id(kwarg)]) for kwarg in task.runnable.kwargs.items()] for task in schedule.tasks)
        ).close()
        self._cx.executemany(
            '''
            INSERT INTO tasks VALUES(?, ?, (SELECT id FROM functions where hash=?))
            ''',
            ((task.id, task.queue.id, hash_cache[id(task.runnable.func)]) for task in schedule.tasks)
        ).close()
        for comp_config in schedule.nonempty_compute_configurations():
            self._cx.executemany(
                '''
                INSERT INTO queues VALUES(?, ?, ?, ?, ?, ?, ?)
                ''',
                itertools.chain.from_iterable(
                    (
                        (queue.id, alloc.configuration.id, allocidx, queue.utilized_resources.cpus, queue.utilized_resources.memory, queue.gpu_id, queue.parent_queue.id if queue.parent_queue else None)
                        for queue in alloc.nonempty_queues()
                    )
                    for (allocidx, alloc) in enumerate(comp_config.nonempty_allocations())
                )
            ).close()

    def tasks_for_queue(self, id: int):
        with closing(self._cx.execute(
            '''
            SELECT tasks.id, functions.function from tasks LEFT JOIN functions on tasks.function=functions.id WHERE queue=?
            ''',
            (id,)
        )) as taskcur:
            for (taskid, func) in taskcur:
                runnable = Runnable(pickle.loads(func), (), {})
                with closing(self._cx.execute(
                    '''
                    SELECT arguments.argument, is_kw
                    FROM task_args LEFT JOIN arguments ON task_args.argument=arguments.id
                    WHERE task_args.task=?
                    ''',
                    (taskid,)
                )) as argcur:
                    for (argpkl, is_kw) in argcur:
                        arg = pickle.loads(argpkl)
                        if is_kw:
                            runnable.kwargs[arg[0]] = arg[1]
                        else:
                            runnable.args += (arg,)
                yield runnable
    
    def count_tasks_for_queue(self, id: int):
        with closing(self._cx.execute(
            '''
            SELECT COUNT(*) from tasks WHERE queue=?
            ''',
            (id,)
        )) as cur:
            return cur.fetchone()[0]

    def queues_for_allocation(self, compute_config_id: int, allocation_id: int):
        with closing(self._cx.execute(
            '''
            SELECT id, cpus, memory, gpu_id FROM queues WHERE configuration=? and allocation=?
            ''',
            (compute_config_id, allocation_id)
        )) as cur:
            for (id, cpus, memory, gpu_id) in cur:
                yield DBQueue(id, cpus, memory,  gpu_id)

    def children_for_queue(self, id: int):
        with closing(self._cx.execute(
            '''
            SELECT id, cpus, gpu_id FROM queues WHERE parent=?
            ''',
            (id,)
        )) as cur:
            for (id, cpus, memory, gpu_id) in cur:
                yield DBQueue(id, cpus, memory, gpu_id)
