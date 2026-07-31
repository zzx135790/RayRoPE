"""Evaluation-only recorder for token depth-width observability."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import torch

from tokenmap.models.scene.probabilistic_flag_rope.uncertainty_application import (
    bounded_depth_endpoints,
)
from tokenmap.models.scene.probabilistic_flag_rope.uncertainty_observability import (
    DepthObservabilityRequest,
)


COUNTERFACTUAL_MODES = (
    "zero",
    "half",
    "double",
    "permute_within_camera",
)
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def depth_counterfactuals(
    predicted_d: torch.Tensor, num_patches: int
) -> dict[str, torch.Tensor]:
    """Build dose controls plus the exact current camera-local roll intervention."""

    if predicted_d.ndim != 3 or predicted_d.shape[-1] != 2:
        raise ValueError("predicted depth must have shape [B,N,2]")
    if num_patches < 1 or predicted_d.shape[1] % num_patches:
        raise ValueError("token count must be divisible by positive num_patches")
    result: dict[str, torch.Tensor] = {}
    for name, scale in (("zero", 0.0), ("half", 0.5), ("double", 2.0)):
        value = predicted_d.clone()
        value[..., 1] = value[..., 1] * scale
        result[name] = value
    camera_count = predicted_d.shape[1] // num_patches
    permuted = predicted_d.clone()
    widths = predicted_d[..., 1].reshape(
        predicted_d.shape[0], camera_count, num_patches
    )
    permuted[..., 1] = torch.roll(widths, shifts=1, dims=2).reshape(
        predicted_d.shape[0], -1
    )
    result["permute_within_camera"] = permuted
    return result


def _distribution(values: torch.Tensor) -> dict[str, object]:
    values = values.detach().float().reshape(-1)
    quantiles = torch.quantile(
        values, torch.tensor(QUANTILES, device=values.device)
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
        "quantiles": {
            f"p{int(probability * 100):02d}": float(value.item())
            for probability, value in zip(QUANTILES, quantiles)
        },
    }


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort(stable=True)
    sorted_values = values[order]
    _, inverse, counts = torch.unique(
        sorted_values, return_inverse=True, return_counts=True
    )
    stops = counts.cumsum(dim=0)
    starts = stops - counts
    group_ranks = (starts + stops - 1).to(values.dtype) * 0.5
    ranks = torch.empty_like(values)
    ranks[order] = group_ranks[inverse]
    return ranks


def _rank_correlation(left: torch.Tensor, right: torch.Tensor) -> Optional[float]:
    """Return tie-aware Spearman correlation for two one-dimensional samples."""

    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if (
        left.numel() < 2
        or torch.unique(left).numel() < 2
        or torch.unique(right).numel() < 2
    ):
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = left_rank.square().sum().sqrt() * right_rank.square().sum().sqrt()
    if denominator.item() == 0.0:
        return None
    return float(((left_rank * right_rank).sum() / denominator).item())


def _camera_width_record(
    actual: torch.Tensor,
    counterfactuals: Mapping[str, torch.Tensor],
) -> dict[str, object]:
    center = actual[:, 0]
    raw_width = actual[:, 1]
    effective_width = raw_width.abs()
    near, far = bounded_depth_endpoints(center, effective_width)
    midpoint = 0.5 * (near + far)
    half_width = 0.5 * (far - near)
    interventions: dict[str, object] = {}
    denominator = effective_width.abs().mean().clamp_min(
        torch.finfo(effective_width.dtype).eps
    )
    for mode, value in counterfactuals.items():
        changed = value[:, 1].abs()
        delta = (changed - effective_width).abs()
        changed_near, changed_far = bounded_depth_endpoints(value[:, 0], changed)
        changed_midpoint = 0.5 * (changed_near + changed_far)
        changed_half_width = 0.5 * (changed_far - changed_near)
        interventions[mode] = {
            "mean_abs_log_width_delta": float(delta.mean().item()),
            "normalized_l1": float((delta.mean() / denominator).item()),
            "max_abs_log_width_delta": float(delta.max().item()),
            "fraction_changed": float((delta > 0).float().mean().item()),
            "position_rank_correlation": (
                _rank_correlation(effective_width, changed)
                if mode == "permute_within_camera"
                else None
            ),
            "abs_log_width_delta": _distribution(delta),
            "linear_midpoint_mean_abs_delta": float(
                (changed_midpoint - midpoint).abs().mean().item()
            ),
            "linear_half_width_mean_abs_delta": float(
                (changed_half_width - half_width).abs().mean().item()
            ),
        }
    return {
        "log_center": _distribution(center),
        "raw_log_half_width": _distribution(raw_width),
        "effective_log_half_width": _distribution(effective_width),
        "linear_midpoint": _distribution(midpoint),
        "linear_half_width": _distribution(half_width),
        "interventions": interventions,
    }


class DepthWidthObservabilityRecorder:
    """Join harness metadata and width evidence with FlagRoPE response summaries."""

    def __init__(
        self,
        *,
        num_patches: int,
        family: str,
        strategy: str,
        query_samples: int = 4,
        key_samples_per_camera: int = 4,
    ) -> None:
        if num_patches < 1:
            raise ValueError("num_patches must be positive")
        if query_samples < 1 or key_samples_per_camera < 1:
            raise ValueError("observability sample counts must be positive")
        self.num_patches = num_patches
        self.family = family
        self.strategy = strategy
        self.query_samples = query_samples
        self.key_samples_per_camera = key_samples_per_camera
        self.records: list[dict[str, object]] = []
        self._scene: Optional[dict[str, object]] = None
        self._forward_index: Optional[int] = None
        self._pending: Optional[dict[str, object]] = None

    def begin_scene(self, *, scene_id: str, scene_index: int, mc_index: int) -> None:
        if not scene_id or min(scene_index, mc_index) < 0:
            raise ValueError("scene observability metadata is invalid")
        if self._pending is not None:
            raise RuntimeError("cannot change scene while a layer record is pending")
        self._scene = {
            "scene_id": scene_id,
            "scene_index": scene_index,
            "mc_index": mc_index,
        }
        self._forward_index = None

    def begin_forward(self, target_view_index: int) -> None:
        if self._scene is None or target_view_index < 0:
            raise RuntimeError("begin_scene must precede begin_forward")
        if self._pending is not None:
            raise RuntimeError("cannot change target view while a layer record is pending")
        self._forward_index = target_view_index

    def request(
        self,
        *,
        layer_index: int,
        original_predicted_d: torch.Tensor,
        actual_predicted_d: torch.Tensor,
    ) -> DepthObservabilityRequest:
        if self._scene is None or self._forward_index is None:
            raise RuntimeError("scene and target view must be set before requesting a trace")
        if self._pending is not None:
            raise RuntimeError("previous observability layer was not recorded")
        counterfactuals = depth_counterfactuals(
            actual_predicted_d, self.num_patches
        )
        B, N, _ = actual_predicted_d.shape
        camera_count = N // self.num_patches
        measurements = torch.stack(
            (
                actual_predicted_d,
                *(counterfactuals[mode] for mode in COUNTERFACTUAL_MODES),
            )
        ).detach().float().cpu()
        actual_measurement = measurements[0]
        counterfactual_measurements = {
            mode: measurements[index + 1]
            for index, mode in enumerate(COUNTERFACTUAL_MODES)
        }
        width_cameras = []
        for batch in range(B):
            for camera in range(camera_count):
                start = camera * self.num_patches
                stop = start + self.num_patches
                width_cameras.append(
                    {
                        "batch_index": batch,
                        "camera_index": camera,
                        **_camera_width_record(
                            actual_measurement[batch, start:stop],
                            {
                                mode: value[batch, start:stop]
                                for mode, value in counterfactual_measurements.items()
                            },
                        ),
                    }
                )
        self._pending = {
            **self._scene,
            "target_view_index": self._forward_index,
            "layer_index": layer_index,
            "batch_size": B,
            "camera_count": camera_count,
            "tokens_per_camera": self.num_patches,
            "original_equals_actual": original_predicted_d is actual_predicted_d,
            "width_cameras": width_cameras,
        }
        return DepthObservabilityRequest(
            layer_index=layer_index,
            original_predicted_d=original_predicted_d,
            actual_predicted_d=actual_predicted_d,
            counterfactuals=counterfactuals,
            recorder=self,
            query_samples=self.query_samples,
            key_samples_per_camera=self.key_samples_per_camera,
        )

    def record_layer(self, payload: Mapping[str, object]) -> None:
        if self._pending is None:
            raise RuntimeError("observability response has no pending width record")
        if payload.get("layer_index") != self._pending["layer_index"]:
            raise ValueError("observability response layer does not match its request")
        responses = payload.get("responses")
        if not isinstance(responses, Mapping) or tuple(responses) != COUNTERFACTUAL_MODES:
            raise ValueError("observability response modes are incomplete or reordered")
        self.records.append({**self._pending, "responses": dict(responses)})
        self._pending = None

    def finalize(self, path: Path, *, metadata: Mapping[str, object]) -> None:
        if self._pending is not None:
            raise RuntimeError("cannot finalize with an incomplete layer record")
        if not path.is_absolute():
            raise ValueError("observability output path must be absolute")
        if not self.records:
            raise ValueError("observability produced no layer records")
        layers = sorted({int(record["layer_index"]) for record in self.records})
        payload = {
            "schema_version": 1,
            "experiment": "re10k-depth-width-observability",
            "status": "completed",
            "family": self.family,
            "strategy": self.strategy,
            "counterfactual_modes": list(COUNTERFACTUAL_MODES),
            "query_samples": self.query_samples,
            "key_samples_per_camera": self.key_samples_per_camera,
            "record_count": len(self.records),
            "layer_indices": layers,
            "metadata": dict(metadata),
            "records": self.records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


__all__ = [
    "COUNTERFACTUAL_MODES",
    "DepthWidthObservabilityRecorder",
    "depth_counterfactuals",
]
