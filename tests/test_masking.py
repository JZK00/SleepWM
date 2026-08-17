import torch

from uniphysio_wm.masking import (
    nonempty_modality_subsets,
    random_full_biased_natural_modality_subset,
    random_modality_presence,
    random_natural_modality_subset,
    random_strict_natural_modality_subset,
    random_span_mask,
)


def test_span_mask_shape_and_coverage() -> None:
    mask = random_span_mask(5, 12, 0.25, min_span=2, max_span=4)
    assert mask.shape == (5, 12)
    assert mask.dtype == torch.bool
    assert torch.all(mask.sum(dim=1) >= 3)


def test_modality_dropout_never_drops_everything() -> None:
    present = random_modality_presence(100, 3, 0.8, torch.device("cpu"))
    assert present.shape == (100, 3)
    assert present.any(dim=1).all()


def test_modality_dropout_is_reproducible() -> None:
    first_generator = torch.Generator().manual_seed(9)
    second_generator = torch.Generator().manual_seed(9)
    first = random_modality_presence(32, 3, 0.7, torch.device("cpu"), generator=first_generator)
    second = random_modality_presence(32, 3, 0.7, torch.device("cpu"), generator=second_generator)
    assert torch.equal(first, second)


def test_uniform_subset_respects_natural_availability() -> None:
    natural = torch.tensor([[True, True, True], [True, False, True], [False, True, False]])
    subset = random_natural_modality_subset(natural, torch.Generator().manual_seed(12))
    assert subset.any(dim=1).all()
    assert not (subset & ~natural).any()
    assert subset[2].tolist() == [False, True, False]


def test_strict_subset_drops_a_modality_when_possible() -> None:
    natural = torch.tensor([[True, True, True], [True, False, True], [False, True, False]])
    subset = random_strict_natural_modality_subset(natural, torch.Generator().manual_seed(12))
    assert subset.any(dim=1).all()
    assert not (subset & ~natural).any()
    assert (subset[:2] != natural[:2]).any(dim=1).all()
    assert subset[2].tolist() == [False, True, False]


def test_full_biased_subset_probability_endpoints() -> None:
    natural = torch.ones(16, 3, dtype=torch.bool)
    full = random_full_biased_natural_modality_subset(natural, 1.0)
    strict = random_full_biased_natural_modality_subset(
        natural,
        0.0,
        torch.Generator().manual_seed(17),
    )
    assert torch.equal(full, natural)
    assert strict.any(dim=1).all()
    assert (strict != natural).any(dim=1).all()


def test_three_modalities_have_seven_nonempty_subsets() -> None:
    subsets = nonempty_modality_subsets(("EEG", "ECG", "EMG"))
    assert len(subsets) == 7
    assert ("EEG", "ECG", "EMG") in subsets
