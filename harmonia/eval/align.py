"""Edit-distance alignment of chord sequences (for bar-wise charts without timing).

Costs: identical chord 0, same root (quality differs) ``same_root_cost``, otherwise 1;
insertion / deletion 1. Ties prefer substitution, then deletion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..theory.chord import Chord


def chord_sub_cost(a: Chord, b: Chord, same_root_cost: float = 0.3) -> float:
    if a.harte() == b.harte():
        return 0.0
    if a.root is not None and a.root == b.root:
        return same_root_cost
    return 1.0


def align(ref: Sequence[Chord], est: Sequence[Chord],
          sub: Callable[[Chord, Chord], float] = chord_sub_cost,
          gap: float = 1.0) -> tuple[float, list[tuple[int | None, int | None]]]:
    """Needleman–Wunsch. Returns (cost, path of (ref_index|None, est_index|None))."""
    n, m = len(ref), len(est)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + sub(ref[i - 1], est[j - 1]),
                          D[i - 1][j] + gap, D[i][j - 1] + gap)
    path: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and abs(D[i][j] - (D[i - 1][j - 1] + sub(ref[i - 1], est[j - 1]))) < 1e-9:
            path.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and abs(D[i][j] - (D[i - 1][j] + gap)) < 1e-9:
            path.append((i - 1, None))
            i -= 1
        else:
            path.append((None, j - 1))
            j -= 1
    return D[n][m], path[::-1]


def est_to_ref_map(path: list[tuple[int | None, int | None]], n_est: int) -> list[int | None]:
    """For each est index, the aligned ref index; insertions inherit the previous aligned ref
    (treated as over-segmentation of that reference chord)."""
    out: list[int | None] = [None] * n_est
    last: int | None = None
    for i, j in path:
        if i is not None:
            last = i
        if j is not None:
            out[j] = i if i is not None else last
    # leading insertions: attach to the first aligned ref
    first = next((x for x in out if x is not None), None)
    return [x if x is not None else first for x in out]
