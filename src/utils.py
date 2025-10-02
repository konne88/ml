from typing import Iterator, List, TypeVar

A = TypeVar("A")


def prefixes(seq: List[A]) -> Iterator[List[A]]:
    for i in range(len(seq)):
        yield seq[:i + 1]
