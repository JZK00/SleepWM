from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import torch


def random_span_mask(
    batch_size: int,
    num_tokens: int,
    mask_ratio: float,
    min_span: int = 1,
    max_span: int = 8,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    if num_tokens < 1 or min_span < 1 or max_span < min_span:
        raise ValueError("invalid token or span configuration")
    target = max(1, int(round(num_tokens * mask_ratio)))
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        attempts = 0
        while int(mask[batch_index].sum()) < target and attempts < num_tokens * 8:
            attempts += 1
            span = int(torch.randint(min_span, max_span + 1, (1,), device=device, generator=generator).item())
            span = min(span, num_tokens)
            start = int(
                torch.randint(0, num_tokens - span + 1, (1,), device=device, generator=generator).item()
            )
            mask[batch_index, start : start + span] = True
    return mask


def random_modality_presence(
    batch_size: int,
    modality_count: int,
    drop_probability: float,
    device: torch.device,
    protected_modality: int | None = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must be in [0, 1)")
    present = torch.rand(batch_size, modality_count, device=device, generator=generator) >= drop_probability
    if protected_modality is not None:
        present[:, protected_modality] = True
    empty = ~present.any(dim=1)
    if empty.any():
        fallback = torch.randint(
            0,
            modality_count,
            (int(empty.sum()),),
            device=device,
            generator=generator,
        )
        present[empty] = False
        present[empty, fallback] = True
    return present


def random_natural_modality_subset(
    natural_present: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample uniformly from each row's nonempty naturally available subsets."""

    if natural_present.ndim != 2:
        raise ValueError("natural_present must be [batch, modalities]")
    natural_present = natural_present.to(dtype=torch.bool)
    if not natural_present.any(dim=1).all():
        raise ValueError("every sample must contain at least one natural modality")
    batch_size, modality_count = natural_present.shape
    selected = torch.zeros_like(natural_present)
    pending = torch.arange(batch_size, device=natural_present.device)
    bit_positions = torch.arange(modality_count, device=natural_present.device)
    while len(pending):
        codes = torch.randint(
            1,
            2**modality_count,
            (len(pending), 1),
            device=natural_present.device,
            generator=generator,
        )
        candidates = codes.bitwise_right_shift(bit_positions).bitwise_and(1).bool()
        candidates = candidates & natural_present[pending]
        accepted = candidates.any(dim=1)
        selected[pending[accepted]] = candidates[accepted]
        pending = pending[~accepted]
    return selected


def random_strict_natural_modality_subset(
    natural_present: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample a proper nonempty subset whenever at least two modalities exist."""

    if natural_present.ndim != 2:
        raise ValueError("natural_present must be [batch, modalities]")
    natural_present = natural_present.to(dtype=torch.bool)
    if not natural_present.any(dim=1).all():
        raise ValueError("every sample must contain at least one natural modality")
    batch_size, modality_count = natural_present.shape
    selected = natural_present.clone()
    pending = torch.nonzero(natural_present.sum(dim=1) > 1, as_tuple=False).flatten()
    bit_positions = torch.arange(modality_count, device=natural_present.device)
    while len(pending):
        codes = torch.randint(
            1,
            2**modality_count,
            (len(pending), 1),
            device=natural_present.device,
            generator=generator,
        )
        candidates = codes.bitwise_right_shift(bit_positions).bitwise_and(1).bool()
        candidates = candidates & natural_present[pending]
        accepted = candidates.any(dim=1) & (candidates != natural_present[pending]).any(dim=1)
        selected[pending[accepted]] = candidates[accepted]
        pending = pending[~accepted]
    return selected


def random_full_biased_natural_modality_subset(
    natural_present: torch.Tensor,
    full_modality_probability: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Choose a full-observation batch or a batch of strict nonempty subsets."""

    if not 0.0 <= full_modality_probability <= 1.0:
        raise ValueError("full_modality_probability must be in [0, 1]")
    keep_full = bool(
        torch.rand((), device=natural_present.device, generator=generator).item()
        < full_modality_probability
    )
    if keep_full:
        return natural_present.to(dtype=torch.bool).clone()
    return random_strict_natural_modality_subset(natural_present, generator=generator)


def nonempty_modality_subsets(modalities: Tuple[str, ...]) -> List[Tuple[str, ...]]:
    return [subset for size in range(1, len(modalities) + 1) for subset in combinations(modalities, size)]
