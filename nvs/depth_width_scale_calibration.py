"""Evaluation-only v2 recorder for depth-width source and scale calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import torch

from nvs.depth_width_observability import _distribution, _rank_correlation
from tokenmap.experiments.flag_rope.depth_width_scale_calibration import (
    CALIBRATORS,
    EXPERIMENT,
    SCALES,
    SCHEMA_VERSION,
    WAVELENGTH_MULTIPLIERS,
    arm_name,
    build_arm_specs,
    build_width_specs,
    width_name,
)
from tokenmap.experiments.flag_rope.frequency_preview import (
    default_preview_config,
    normalize_preview_config,
)
from tokenmap.models.scene.probabilistic_flag_rope.frequency_layout import (
    build_layout_banks,
)
from tokenmap.models.scene.probabilistic_flag_rope.uncertainty_application import (
    MAX_LOG_DEPTH,
    bounded_depth_endpoints,
)
from tokenmap.models.scene.probabilistic_flag_rope.uncertainty_observability import (
    DepthObservabilityRequest,
    FrequencyShadowProbe,
)


ARM_SPECS = build_arm_specs()
WIDTH_SPECS = build_width_specs()


def _calibrated_width(
    raw_width: torch.Tensor,
    *,
    calibrator: str,
    scale: float,
    layer_median: float,
) -> torch.Tensor:
    raw = raw_width.abs()
    if calibrator == "raw_multiplier":
        return raw * scale
    if calibrator == "median_unit":
        if not math.isfinite(layer_median) or layer_median <= 0.0:
            raise ValueError("median_unit requires a finite positive train-layer median")
        return raw / layer_median * scale
    raise ValueError(f"unsupported depth-width calibrator: {calibrator}")


def build_width_counterfactuals(
    predicted_d: torch.Tensor, *, layer_median: float
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for calibrator in CALIBRATORS:
        for scale in SCALES:
            changed = predicted_d.clone()
            changed[..., 1] = _calibrated_width(
                predicted_d[..., 1],
                calibrator=calibrator,
                scale=scale,
                layer_median=layer_median,
            )
            result[width_name(calibrator, scale)] = changed
    return result


def _target_frequency_banks(
    family: str, multiplier: int
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[str, ...]]]:
    if multiplier not in WAVELENGTH_MULTIPLIERS:
        raise ValueError("wavelength multiplier is outside the locked grid")
    arm = "dual_sinc" if family == "ray_point" else "segment_bounds"
    banks = build_layout_banks(normalize_preview_config(default_preview_config(arm)))
    if family == "ray_point":
        bank = banks["point"]
        return (
            {"point": torch.from_numpy(bank.omega).float() / multiplier},
            {"point": tuple(bank.head_groups)},
        )
    if family == "segment_bounds":
        bank = banks["segment"]
        omega = torch.from_numpy(bank.omega).float()
        omega[..., 7] /= multiplier
        return (
            {"segment_bounds": omega},
            {"segment_bounds": tuple(bank.head_groups)},
        )
    raise ValueError("target frequency shadows require ray_point or segment_bounds")


def _width_diagnostics(
    original: torch.Tensor, changed: torch.Tensor
) -> dict[str, object]:
    raw = original[..., 1].abs().detach().float().cpu().reshape(-1)
    width = changed[..., 1].abs().detach().float().cpu().reshape(-1)
    center = changed[..., 0].detach().float().cpu().reshape(-1)
    zero = torch.zeros_like(width)
    near, far = bounded_depth_endpoints(center, width)
    zero_near, zero_far = bounded_depth_endpoints(center, zero)
    min_log = math.log(torch.finfo(center.dtype).tiny)
    near_clamp = (center - width <= min_log).float().mean()
    far_clamp = (center + width >= MAX_LOG_DEPTH).float().mean()
    zero_near_clamp = (center <= min_log).float().mean()
    zero_far_clamp = (center >= MAX_LOG_DEPTH).float().mean()
    return {
        "raw_log_half_width": _distribution(raw),
        "calibrated_log_half_width": _distribution(width),
        "rank_correlation_vs_raw": _rank_correlation(raw, width),
        "near_clamp_fraction": float(near_clamp.item()),
        "far_clamp_fraction": float(far_clamp.item()),
        "near_clamp_increment": float((near_clamp - zero_near_clamp).item()),
        "far_clamp_increment": float((far_clamp - zero_far_clamp).item()),
        "linear_near": _distribution(near),
        "linear_far": _distribution(far),
        "zero_linear_near": _distribution(zero_near),
        "zero_linear_far": _distribution(zero_far),
    }


class DepthWidthScaleCalibrationRecorder:
    """Collect source widths or issue formal paired scale/frequency probes."""

    def __init__(
        self,
        *,
        num_patches: int,
        family: str,
        strategy: str,
        stage: str,
        layer_medians: Optional[Mapping[str, float]] = None,
        query_samples: int = 4,
        key_samples_per_camera: int = 4,
    ) -> None:
        if stage not in ("source_audit", "selection_evaluation"):
            raise ValueError("invalid scale-calibration stage")
        if stage == "selection_evaluation" and family not in (
            "ray_point", "segment_bounds"
        ):
            raise ValueError("formal scale calibration requires target-sensitive family")
        self.num_patches = num_patches
        self.family = family
        self.strategy = strategy
        self.stage = stage
        self.layer_medians = dict(layer_medians or {})
        self.query_samples = query_samples
        self.key_samples_per_camera = key_samples_per_camera
        self.records: list[dict[str, object]] = []
        self._scene: Optional[dict[str, object]] = None
        self._forward_index: Optional[int] = None
        self._pending: Optional[dict[str, object]] = None

    def begin_scene(
        self,
        *,
        scene_id: str,
        scene_index: int,
        mc_index: int,
        context_baseline_max: float,
    ) -> None:
        if not scene_id or min(scene_index, mc_index) < 0:
            raise ValueError("scale-calibration scene metadata is invalid")
        if self._pending is not None:
            raise RuntimeError("cannot change scene with a pending formal layer")
        self._scene = {
            "scene_id": scene_id,
            "scene_index": scene_index,
            "mc_index": mc_index,
            "context_baseline_max": float(context_baseline_max),
        }
        self._forward_index = None

    def begin_forward(self, target_view_index: int) -> None:
        if self._scene is None or target_view_index < 0:
            raise RuntimeError("begin_scene must precede begin_forward")
        self._forward_index = target_view_index

    def request(
        self,
        *,
        layer_index: int,
        original_predicted_d: torch.Tensor,
        actual_predicted_d: torch.Tensor,
    ) -> Optional[DepthObservabilityRequest]:
        if self._scene is None or self._forward_index is None:
            raise RuntimeError("scene and target view must precede scale calibration")
        if original_predicted_d is not actual_predicted_d:
            raise ValueError("scale calibration must start from the true depth tensor")
        B, N, _ = actual_predicted_d.shape
        C = N // self.num_patches
        base = {
            **self._scene,
            "target_view_index": self._forward_index,
            "layer_index": layer_index,
            "batch_size": B,
            "camera_count": C,
            "tokens_per_camera": self.num_patches,
        }
        if self.stage == "source_audit":
            widths = actual_predicted_d[..., 1].detach().float().abs().cpu().reshape(-1)
            self.records.append({
                **base,
                "raw_log_half_width": [float(value) for value in widths.tolist()],
                "raw_log_half_width_distribution": _distribution(widths),
            })
            return None
        if self._pending is not None:
            raise RuntimeError("previous formal scale-calibration layer was not recorded")
        median = float(self.layer_medians.get(str(layer_index), float("nan")))
        widths = build_width_counterfactuals(
            actual_predicted_d, layer_median=median
        )
        zero = widths[width_name("raw_multiplier", 0.0)]
        references = {
            arm_name(calibrator, scale, 1): zero
            for calibrator in CALIBRATORS for scale in SCALES
        }
        counterfactuals = {
            arm_name(calibrator, scale, 1): widths[width_name(calibrator, scale)]
            for calibrator in CALIBRATORS for scale in SCALES
        }
        shadows = []
        for multiplier in WAVELENGTH_MULTIPLIERS[1:]:
            omega, groups = _target_frequency_banks(self.family, multiplier)
            shadow_references = {
                arm_name(calibrator, scale, multiplier): zero
                for calibrator in CALIBRATORS for scale in SCALES
            }
            shadow_counterfactuals = {
                arm_name(calibrator, scale, multiplier): widths[
                    width_name(calibrator, scale)
                ]
                for calibrator in CALIBRATORS for scale in SCALES
            }
            shadows.append(FrequencyShadowProbe(
                name=f"wavelength_x{multiplier}",
                omega_overrides={key: value.to(actual_predicted_d) for key, value in omega.items()},
                head_groups_by_family=groups,
                comparison_references=shadow_references,
                counterfactuals=shadow_counterfactuals,
            ))
        self._pending = {
            **base,
            "layer_median": median,
            "width_diagnostics": {
                name: _width_diagnostics(actual_predicted_d, changed)
                for name, changed in widths.items()
            },
        }
        return DepthObservabilityRequest(
            layer_index=layer_index,
            original_predicted_d=original_predicted_d,
            actual_predicted_d=actual_predicted_d,
            counterfactuals=counterfactuals,
            comparison_references=references,
            shadow_frequencies=tuple(shadows),
            recorder=self,
            query_samples=self.query_samples,
            key_samples_per_camera=self.key_samples_per_camera,
            compact_phase=True,
        )

    def record_layer(self, payload: Mapping[str, object]) -> None:
        if self._pending is None:
            raise RuntimeError("formal scale-calibration response has no pending layer")
        if payload.get("layer_index") != self._pending["layer_index"]:
            raise ValueError("formal scale-calibration layer identity changed")
        responses = payload.get("responses")
        shadows = payload.get("shadow_responses")
        if not isinstance(responses, Mapping) or not isinstance(shadows, Mapping):
            raise ValueError("formal scale-calibration response is incomplete")
        self.records.append({
            **self._pending,
            "generator_state_sha256": payload["generator_state_sha256"],
            "responses": dict(responses),
            "shadow_responses": dict(shadows),
        })
        self._pending = None

    def finalize(self, path: Path, *, metadata: Mapping[str, object]) -> None:
        if self._pending is not None or not self.records:
            raise RuntimeError("cannot finalize incomplete scale-calibration evidence")
        if not path.is_absolute():
            raise ValueError("scale-calibration output path must be absolute")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "completed",
            "stage": self.stage,
            "family": self.family,
            "strategy": self.strategy,
            "arm_specs": ARM_SPECS if self.stage == "selection_evaluation" else {},
            "width_specs": WIDTH_SPECS if self.stage == "selection_evaluation" else {},
            "query_samples": self.query_samples,
            "key_samples_per_camera": self.key_samples_per_camera,
            "record_count": len(self.records),
            "layer_indices": sorted({int(row["layer_index"]) for row in self.records}),
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
    "ARM_SPECS",
    "DepthWidthScaleCalibrationRecorder",
    "WIDTH_SPECS",
    "build_width_counterfactuals",
]
