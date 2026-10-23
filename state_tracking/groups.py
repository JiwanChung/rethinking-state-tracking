"""Finite groups stored as Cayley tables.

table[a, b] = a * b. For S3 the product is composition, (p1 * p2)(i) = p1(p2(i)),
over the permutations of {0, 1, 2} in sorted order.
"""

from itertools import permutations

import numpy as np


class FiniteGroup:
    def __init__(self, table, name):
        self.table = table.astype(np.int64)
        self.order = table.shape[0]
        self.name = name

    def multiply(self, a, b):
        return int(self.table[a, b])

    def cumulative_product(self, seq):
        """Running product y_t = x_1 * x_2 * ... * x_t of a (T,) sequence."""
        T = len(seq)
        result = np.empty(T, dtype=np.int64)
        acc = int(seq[0])
        result[0] = acc
        for t in range(1, T):
            acc = self.multiply(acc, int(seq[t]))
            result[t] = acc
        return result

    def __repr__(self):
        return f"FiniteGroup({self.name}, order={self.order})"


def cyclic_group(n):
    table = np.zeros((n, n), dtype=np.int64)
    for a in range(n):
        for b in range(n):
            table[a, b] = (a + b) % n
    return FiniteGroup(table, f"C{n}")


def _compose_perms(p1, p2):
    return tuple(p1[p2[i]] for i in range(len(p1)))


def symmetric_group(n):
    all_perms = sorted(permutations(range(n)))
    order = len(all_perms)
    perm_index = {p: i for i, p in enumerate(all_perms)}
    table = np.zeros((order, order), dtype=np.int64)
    for i, p1 in enumerate(all_perms):
        for j, p2 in enumerate(all_perms):
            table[i, j] = perm_index[_compose_perms(p1, p2)]
    return FiniteGroup(table, f"S{n}")


GROUPS = {
    "C2": lambda: cyclic_group(2),
    "C6": lambda: cyclic_group(6),
    "S3": lambda: symmetric_group(3),
}
