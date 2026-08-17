from __future__ import annotations

from types import SimpleNamespace

from train_external_transfer import select_subject_fraction


def test_subject_fraction_selects_complete_subject_records() -> None:
    dataset = SimpleNamespace(
        records=[
            {"subject": "a"},
            {"subject": "a"},
            {"subject": "b"},
            {"subject": "c"},
        ],
        index=[(0, 0), (0, 1), (1, 0), (2, 0), (3, 0)],
        transition_mask=[(False,), (False,), (True,), (False,), (True,)],
        transition_window=[False, False, True, False, True],
    )
    subset, subjects = select_subject_fraction(dataset, fraction=0.34, seed=7)
    selected_record_subjects = {
        dataset.records[record_index]["subject"] for record_index, _ in subset.index
    }
    assert selected_record_subjects == set(subjects)
    assert len(subjects) == 2


def test_full_fraction_keeps_every_subject_and_sequence() -> None:
    dataset = SimpleNamespace(
        records=[{"subject": "a"}, {"subject": "b"}],
        index=[(0, 0), (0, 1), (1, 0)],
        transition_mask=[(False,), (True,), (False,)],
        transition_window=[False, True, False],
    )
    subset, subjects = select_subject_fraction(dataset, fraction=1.0, seed=7)
    assert subjects == ["a", "b"]
    assert subset.index == [(0, 0), (0, 1), (1, 0)]
