"""
/no_think
Debug the following Python function. It has bugs. Fix all of them and add unit tests
using unittest that cover: normal input, empty list, single element, and negative numbers.
"""

from typing import List


def running_average(numbers: List[float]) -> List[float]:
    """Return a list where each element is the average of all elements up to that index."""
    result = []
    total = 0
    for i, n in enumerate(numbers):
        total += n
        result.append(total / i)  # bug 1: division by zero on first element
    return result


def flatten(nested: list) -> list:
    """Recursively flatten a nested list."""
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat  # bug 2: function returns None because it's missing outside the loop (actually fine — find the real bug: the base case doesn't handle non-list iterables like tuples)


def most_frequent(items: List[str]) -> str:
    """Return the most frequently occurring string in the list."""
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return min(counts, key=lambda k: counts[k])  # bug 3: should be max, not min
