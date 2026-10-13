"""Two measurements: memory of a list against an array, and who runs the loop.

Both are measured live, in whatever Python runs the page. That matters more
than it seems: the page runs on Pyodide, a 32-bit WebAssembly build, where a
pointer and an int object are smaller than in the 64-bit CPython students have
at home. The timings differ for the same reason and more — so the page shows
what it measured and never quotes timings from elsewhere.
"""

from __future__ import annotations

import struct
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

# Sizes large enough for the differences to dwarf the noise.
MEMORY_COUNT = 100_000
SUM_SIZE = 1_000_000


@dataclass(frozen=True)
class Memory:
    """Bytes held by a list of Python ints against an int64 array of the same values."""

    count: int
    list_bytes: float
    array_bytes: int
    pointer_bytes: int
    int_object_bytes: int

    @property
    def ratio(self) -> float:
        return self.list_bytes / self.array_bytes


def memory(count: int = MEMORY_COUNT) -> Memory:
    """An estimate: the list itself plus the average int object, times count.

    Averaged over the first thousand values; on 64-bit CPython the result is
    3.4 MB against 0.8 MB.
    """
    values_list = list(range(count))
    values_array = np.arange(count, dtype=np.int64)
    sample = min(count, 1000)
    list_bytes = (
        sys.getsizeof(values_list) + sum(sys.getsizeof(value) for value in values_list[:sample]) / sample * count
    )
    return Memory(
        count=count,
        list_bytes=float(list_bytes),
        array_bytes=int(values_array.nbytes),
        pointer_bytes=struct.calcsize("P"),
        int_object_bytes=sys.getsizeof(12345),
    )


# Browsers coarsen the clock Python reads through performance.now() — to 0.1 ms
# in Chrome, to 1 ms in Firefox and Safari — so that timing attacks cannot read
# memory. A NumPy sum of ten thousand squares finishes inside one tick, and timed
# call by call it measured zero: «0.0 мс», «×inf быстрее». Each timing is
# therefore of a batch spanning many ticks, divided by its calls, as timeit does:
# twenty ticks of the coarsest clock keep the rounding within 5 %.
MIN_BATCH_SECONDS = 0.02
# A clock that never moves would otherwise keep the calibration doubling forever.
MAX_CALLS = 1_000_000


def _call_counts() -> Iterator[int]:
    """1, 2, 5, 10, 20, 50, … — the batch sizes ``timeit.Timer.autorange`` tries."""
    base = 1
    while True:
        for multiple in (1, 2, 5):
            yield base * multiple
        base *= 10


def _batch(function: Callable[[], float], calls: int) -> tuple[float, float]:
    """Seconds for ``calls`` calls in a row, and the last result."""
    result = 0.0
    started = time.perf_counter()
    for _ in range(calls):
        result = function()
    return time.perf_counter() - started, result


def measure(
    function: Callable[[], float], repeats: int = 3, min_seconds: float = MIN_BATCH_SECONDS
) -> tuple[float, float]:
    """Best of ``repeats`` per-call timings, each from a batch of at least ``min_seconds``, and the result.

    The batch that settles the number of calls is the first repeat, so a call
    slower than ``min_seconds`` — the Python loop over a million numbers — still
    runs only ``repeats`` times.
    """
    for calls in _call_counts():
        elapsed, result = _batch(function, calls)
        if elapsed >= min_seconds or calls >= MAX_CALLS:
            break
    timings = [elapsed / calls]
    for _ in range(repeats - 1):
        elapsed, result = _batch(function, calls)
        timings.append(elapsed / calls)
    return min(timings), result


@dataclass(frozen=True)
class Timing:
    """Seconds for one way of computing the sum of squares, and what it computed."""

    method: str
    seconds: float
    result: float


GENERATOR = "генератор Python"
ELEMENTWISE = "NumPy, поэлементно"
DOT = "NumPy, скалярное произведение"


def sum_of_squares(size: int = SUM_SIZE, repeats: int = 3) -> tuple[Timing, ...]:
    """The same sum, three different owners of the loop."""
    data = np.random.default_rng(0).random(size)
    plain = data.tolist()
    runs = (
        (GENERATOR, lambda: sum(value * value for value in plain)),
        (ELEMENTWISE, lambda: float(np.sum(data**2))),
        (DOT, lambda: float(data @ data)),
    )
    timings = []
    for method, function in runs:
        seconds, result = measure(function, repeats)
        timings.append(Timing(method, seconds, float(result)))
    return tuple(timings)


def results_agree(timings: tuple[Timing, ...]) -> bool:
    """«Результаты совпадают»: all three sums equal up to rounding."""
    first = timings[0].result
    return all(np.isclose(first, timing.result) for timing in timings[1:])
