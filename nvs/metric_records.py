"""JSON-ready per-scene metric records for paired experiment statistics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def build_scene_metric_record(
    target_paths,
    *,
    psnr,
    ssim,
    lpips,
) -> dict[str, float | str]:
    """Build one stable scene record from a batch-one evaluation item."""
    flattened = np.asarray(target_paths, dtype=object).reshape(-1)
    if flattened.size == 0:
        raise ValueError("target_paths must contain at least one path")
    path = Path(str(flattened[0]))
    if len(path.parents) < 2:
        raise ValueError(f"target path does not identify a scene directory: {path}")
    return {
        "scene_id": path.parents[1].name,
        "psnr": _finite_scalar(psnr, "psnr"),
        "ssim": _finite_scalar(ssim, "ssim"),
        "lpips": _finite_scalar(lpips, "lpips"),
    }


def merge_rank_metric_records(
    records_by_rank: Iterable[Iterable[dict[str, object]]],
) -> list[dict[str, object]]:
    """Flatten distributed records, rejecting ambiguous duplicate scenes."""
    merged = [record for rank_records in records_by_rank for record in rank_records]
    scene_ids = [record.get("scene_id") for record in merged]
    if any(not isinstance(scene_id, str) or not scene_id for scene_id in scene_ids):
        raise ValueError("every metric record requires a non-empty scene_id")
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("duplicate scene_id in distributed metric records")
    return sorted(merged, key=lambda record: str(record["scene_id"]))


def build_metrics_payload(
    *,
    label: str,
    step: int,
    n_total: int,
    psnr,
    ssim,
    lpips,
    records_by_rank: Iterable[Iterable[dict[str, object]]],
) -> dict[str, object]:
    """Combine aggregate metrics with the exact paired scene observations."""
    per_scene = merge_rank_metric_records(records_by_rank)
    if len(per_scene) != n_total:
        raise ValueError(
            f"n_total ({n_total}) does not match per_scene count ({len(per_scene)})"
        )
    return {
        "label": label,
        "step": int(step),
        "n_total": int(n_total),
        "psnr": _finite_scalar(psnr, "psnr"),
        "ssim": _finite_scalar(ssim, "ssim"),
        "lpips": _finite_scalar(lpips, "lpips"),
        "per_scene": per_scene,
    }


def _finite_scalar(value, field: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field} must be scalar")
        value = value.detach().cpu().item()
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed
