from __future__ import annotations

import torch

from evaluate_belief_outcomes import (
    rankdata,
    select_f1_threshold,
    subject_calibration_mask,
)


def test_rankdata_averages_ties() -> None:
    values = rankdata(torch.tensor([3.0, 1.0, 1.0, 2.0]).numpy())
    assert values.tolist() == [4.0, 1.5, 1.5, 3.0]


def test_subject_split_keeps_subjects_together() -> None:
    subjects = torch.tensor([1, 1, 2, 2, 3, 3]).numpy().astype(str)
    mask = subject_calibration_mask(subjects)
    assert 0 < int(mask.sum()) < len(mask)
    assert mask[0] == mask[1]
    assert mask[2] == mask[3]
    assert mask[4] == mask[5]


def test_fast_threshold_selects_best_observed_f1() -> None:
    target = torch.tensor([0, 1, 1, 0, 1], dtype=torch.bool).numpy()
    score = torch.tensor([0.1, 0.8, 0.7, 0.6, 0.5]).numpy()
    assert select_f1_threshold(target, score) == 0.5
