"""TPR/FPR trajectories and normalized AUC (paper §6.3).

Labels are per-run (did the agent ever violate). Trajectory points are (FPR, TPR)
after each round, starting at (0, 0). Normalized AUC = trapezoidal area under the
trajectory / area under the diagonal over the same FPR range (= fpr_max^2 / 2).
"""

import math

Point = tuple[float, float]


def _rate(count: int, total: int) -> float:
    return count / total if total else math.nan


def tpr_fpr_trajectory(
    labels: dict[int, bool], removed_round: dict[int, int], num_rounds: int
) -> list[Point]:
    violators = [a for a, y in labels.items() if y]
    compliant = [a for a, y in labels.items() if not y]
    points: list[Point] = [(0.0 if compliant else math.nan, 0.0 if violators else math.nan)]
    for t in range(1, num_rounds + 1):
        tp = sum(1 for a in violators if removed_round.get(a, math.inf) <= t)
        fp = sum(1 for a in compliant if removed_round.get(a, math.inf) <= t)
        points.append((_rate(fp, len(compliant)), _rate(tp, len(violators))))
    return points


def normalized_auc(trajectory: list[Point]) -> float:
    if any(math.isnan(x) or math.isnan(y) for x, y in trajectory):
        return math.nan
    fpr_max = max(x for x, _ in trajectory)
    if fpr_max <= 0:
        return math.nan
    area = sum(
        (x1 - x0) * (y0 + y1) / 2
        for (x0, y0), (x1, y1) in zip(trajectory, trajectory[1:])
    )
    return area / (fpr_max**2 / 2)


def is_defined(trajectory: list[Point]) -> bool:
    return not any(math.isnan(x) or math.isnan(y) for x, y in trajectory)


def mean_trajectory(trajectories: list[list[Point]]) -> list[Point]:
    """Pointwise mean over seeds whose trajectory is fully defined (both FPR and TPR at every round).

    A run with no compliant (or no violating) agents has an undefined FPR (or TPR) throughout; it is
    excluded as a whole, so its other coordinate cannot leak into the average. Runs must share length.
    """
    usable = [t for t in trajectories if is_defined(t)]
    if not usable:
        return [(math.nan, math.nan)] * len(trajectories[0])
    n = len(usable)
    return [
        (sum(t[i][0] for t in usable) / n, sum(t[i][1] for t in usable) / n)
        for i in range(len(usable[0]))
    ]
