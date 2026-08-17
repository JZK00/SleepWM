from __future__ import annotations

import numpy as np
import torch

from evaluate_sleep_events import event_risk_scores, next_event_offset, select_f1_threshold


def test_next_event_offset_uses_adjacent_stage_onset() -> None:
    labels = np.asarray([0, 0, 1, 2, 2, 4, 4])
    assert next_event_offset(labels, 0, "transition", 6) == 2
    assert next_event_offset(labels, 0, "sleep_onset", 6) == 2
    assert next_event_offset(labels, 2, "rem_onset", 4) == 3


def test_event_risks_are_cumulative_across_horizons() -> None:
    current = torch.tensor([[0.8, 0.1, 0.1, 0.0, 0.0]])
    future = torch.tensor([[[0.7, 0.2, 0.1, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0, 0.0]]])
    scores = event_risk_scores(current, future)
    assert scores["transition"][0, 1] >= scores["transition"][0, 0]
    assert scores["sleep_onset"][0, 1] >= scores["sleep_onset"][0, 0]


def test_threshold_selection_separates_simple_scores() -> None:
    target = np.asarray([False, False, True, True])
    score = np.asarray([0.1, 0.2, 0.8, 0.9])
    threshold = select_f1_threshold(target, score)
    assert 0.2 < threshold <= 0.8
