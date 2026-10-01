"""Lecture 2's two measurements: memory of a list against an array, and who runs the loop.

Both are measured live, in whatever Python runs the page. That matters more
than it seems: the lecture's numbers come from 64-bit CPython, while the page
runs on Pyodide, a 32-bit WebAssembly build, where a pointer and an int object
are smaller. The timings differ for the same reason and more — so the page
shows what it measured and never claims the lecture's timings as its own.
"""

from __future__ import annotations

import struct
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Lecture 2's sizes.
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
    """The lecture's estimate: the list itself plus the average int object, times count.

    Averaged over the first thousand values, exactly as the lecture does, so on
    64-bit CPython the result is the lecture's «3.4 МБ» against «0.8 МБ».
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


def measure(function: Callable[[], float], repeats: int = 3) -> tuple[float, float]:
    """Best of ``repeats`` wall-clock timings, and the result. The lecture's helper."""
    timings = []
    result = 0.0
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
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
    """Lecture 2's comparison: the same sum, three different owners of the loop."""
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
    """The lecture's «Результаты совпадают»: all three sums equal up to rounding."""
    first = timings[0].result
    return all(np.isclose(first, timing.result) for timing in timings[1:])
