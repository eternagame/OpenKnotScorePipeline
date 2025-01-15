'''
Provides a mechanism for tasks to defer some action such that it can be performed in
bulk after multiple tasks have run
'''

from typing import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

@dataclass
class Deferred:
    func: Callable[[], None]
    flush_after_count: int | None = None
    flush_after_time: timedelta | None = None

    count_since_flushed: int = 0
    last_flushed_time: datetime = field(default_factory=datetime.now)

registry = dict[int, Deferred]()

def register_deferred(deferred: Deferred):
    registry[id(deferred)] = deferred
    deferred.count_since_flushed += 1
    
    if (
        (
            deferred.flush_after_count is not None
            and deferred.count_since_flushed > deferred.flush_after_count
        )
        or (
            deferred.flush_after_time is not None
            and datetime.now() - deferred.last_flushed_time > deferred.flush_after_time
        )
    ):
        deferred.func()

def flush_deferred():
    while registry:
        registry.popitem()[1].func()
