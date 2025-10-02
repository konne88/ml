from dataclasses import dataclass
import math
from typing import Iterator, List, TypeVar

Token = int


Element = TypeVar("Element")


def normalize(list: List[float]) -> List[float]:
    total = sum(list)
    return [element / total for element in list]


def weighted_average(weights: List[float], values):
    return sum([weight * value for weight, value in zip(weights, values)])


A = TypeVar("A")


def prefixes(seq: List[A]) -> Iterator[List[A]]:
    for i in range(len(seq)):
        yield seq[:i + 1]
