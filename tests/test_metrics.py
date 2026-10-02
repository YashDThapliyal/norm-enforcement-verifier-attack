import math

import pytest

from nem.metrics import normalized_auc, tpr_fpr_trajectory


def test_trajectory_counts_cumulative_removals():
    labels = {0: True, 1: True, 2: False, 3: False}
    removed_round = {0: 2, 2: 3}
    traj = tpr_fpr_trajectory(labels, removed_round, num_rounds=4)
    assert traj == [(0.0, 0.0), (0.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.5, 0.5)]


def test_auc_diagonal_is_one():
    traj = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5)]
    assert normalized_auc(traj) == pytest.approx(1.0)


def test_auc_good_trajectory_above_one():
    traj = [(0.0, 0.0), (0.0, 0.75), (0.5, 1.0)]
    # area = 0.5 * (0.75 + 1.0) / 2 = 0.4375; diagonal = 0.125
    assert normalized_auc(traj) == pytest.approx(3.5)


def test_auc_empty_fpr_range_is_nan():
    traj = [(0.0, 0.0), (0.0, 0.5)]
    assert math.isnan(normalized_auc(traj))


def test_trajectory_nan_when_no_violators():
    traj = tpr_fpr_trajectory({0: False, 1: False}, {0: 1}, num_rounds=2)
    assert all(math.isnan(t) for _, t in traj)


def test_mean_trajectory_drops_runs_with_undefined_points_jointly():
    from nem.metrics import mean_trajectory

    defined = [(0.0, 0.0), (0.5, 1.0)]
    no_compliant = [(math.nan, 0.0), (math.nan, 0.0)]  # FPR undefined: its TPR must not leak in
    assert mean_trajectory([defined, no_compliant]) == defined


def test_mean_trajectory_all_undefined_is_nan():
    from nem.metrics import mean_trajectory

    out = mean_trajectory([[(math.nan, 0.0)]])
    assert math.isnan(out[0][0]) and math.isnan(out[0][1])
