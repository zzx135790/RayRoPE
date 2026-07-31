"""Evaluation-only depth-width routing, dose, and shadow-capacity recorder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Optional

import torch

from nvs.depth_width_observability import _distribution, _rank_correlation
from tokenmap.experiments.flag_rope.depth_width_calibration import (
    EXPERIMENT,
    SCHEMA_VERSION,
    SHADOW_COMPARISONS,
    build_arm_specs,
    build_capacity_manifest,
    build_shadow_banks,
)
from tokenmap.models.scene.probabilistic_flag_rope.uncertainty_application import (
    MAX_LOG_DEPTH,
    bounded_depth_endpoints,
)
from tokenmap.models.scene.probabilistic_flag_rope.uncertainty_observability import (
    DepthObservabilityRequest,
    SegmentShadowProbe,
)


ARM_SPECS = build_arm_specs()
COUNTERFACTUAL_MODES = tuple(ARM_SPECS)


def _dose(
    predicted_d: torch.Tensor, *, scale: float, cap: float | None
) -> torch.Tensor:
    result = predicted_d.clone()
    width = predicted_d[..., 1].abs() * scale
    if cap is not None:
        width = width.clamp(max=cap)
    result[..., 1] = width
    return result


def _routing_seed(
    scene_id: str, layer_index: int, batch_index: int, camera_index: int
) -> int:
    payload = (
        f"depth-width-calibration-v1:{scene_id}:{layer_index}:"
        f"{batch_index}:{camera_index}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _route(
    predicted_d: torch.Tensor,
    *,
    routing: str,
    num_patches: int,
    scene_id: str,
    layer_index: int,
) -> torch.Tensor:
    if routing not in ("roll", "random", "reverse"):
        raise ValueError(f"unsupported routing intervention: {routing}")
    result = predicted_d.clone()
    B, N, _ = result.shape
    if N % num_patches:
        raise ValueError("token count must be divisible by num_patches")
    C = N // num_patches
    widths = result[..., 1].reshape(B, C, num_patches)
    changed = torch.empty_like(widths)
    for batch in range(B):
        for camera in range(C):
            current = widths[batch, camera]
            if routing == "roll":
                changed[batch, camera] = current.roll(1)
            elif routing == "random":
                generator = torch.Generator(device="cpu")
                generator.manual_seed(
                    _routing_seed(scene_id, layer_index, batch, camera)
                )
                order = torch.randperm(num_patches, generator=generator).to(
                    current.device
                )
                changed[batch, camera] = current[order]
            else:
                order = current.abs().argsort(stable=True)
                reversed_values = current[order].flip(0)
                changed[batch, camera, order] = reversed_values
    result[..., 1] = changed.reshape(B, N)
    return result


def build_calibration_comparisons(
    predicted_d: torch.Tensor,
    *,
    num_patches: int,
    scene_id: str,
    layer_index: int,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[tuple[float, float | None], torch.Tensor],
]:
    """Build paired references/counterfactuals with shared tensor identities."""

    dose_cache: dict[tuple[float, float | None], torch.Tensor] = {}

    def dose(scale: float, cap: float | None) -> torch.Tensor:
        key = (scale, cap)
        if key not in dose_cache:
            dose_cache[key] = _dose(predicted_d, scale=scale, cap=cap)
        return dose_cache[key]

    zero = dose(0.0, None)
    references: dict[str, torch.Tensor] = {}
    counterfactuals: dict[str, torch.Tensor] = {}
    for name, spec in ARM_SPECS.items():
        scale = float(spec["scale"])
        cap = None if spec["cap"] is None else float(spec["cap"])
        correct = dose(scale, cap)
        if spec["kind"] == "dose":
            references[name] = zero
            counterfactuals[name] = correct
        else:
            references[name] = correct
            counterfactuals[name] = _route(
                correct,
                routing=str(spec["routing"]),
                num_patches=num_patches,
                scene_id=scene_id,
                layer_index=layer_index,
            )
    return references, counterfactuals, dose_cache


def _multiset_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.sort().values, right.sort().values))


def _comparison_record(
    original: torch.Tensor,
    reference: torch.Tensor,
    changed: torch.Tensor,
    spec: Mapping[str, object],
) -> dict[str, object]:
    original_width = original[:, 1].abs()
    reference_width = reference[:, 1].abs()
    changed_width = changed[:, 1].abs()
    input_delta = (changed_width - reference_width).abs()
    original_near, original_far = bounded_depth_endpoints(
        original[:, 0], original_width
    )
    reference_near, reference_far = bounded_depth_endpoints(
        reference[:, 0], reference_width
    )
    changed_near, changed_far = bounded_depth_endpoints(
        changed[:, 0], changed_width
    )
    original_half = 0.5 * (original_far - original_near)
    reference_half = 0.5 * (reference_far - reference_near)
    changed_half = 0.5 * (changed_far - changed_near)
    endpoint_delta = (changed_half - reference_half).abs()
    eps = torch.finfo(original_width.dtype).eps
    cap = spec.get("cap")
    scaled = original_width * float(spec["scale"])
    return {
        "fraction_changed": float((input_delta > 0).float().mean().item()),
        "mean_abs_log_width_delta": float(input_delta.mean().item()),
        "normalized_l1_reference": float(
            (input_delta.mean() / reference_width.mean().clamp_min(eps)).item()
        ),
        "normalized_l1_actual": float(
            (input_delta.mean() / original_width.mean().clamp_min(eps)).item()
        ),
        "position_rank_correlation": _rank_correlation(
            reference_width, changed_width
        ),
        "rank_correlation_vs_actual": _rank_correlation(
            original_width, changed_width
        ),
        "width_multiset_preserved": (
            _multiset_equal(reference[:, 1], changed[:, 1])
            if spec["kind"] == "routing"
            else None
        ),
        "linear_half_width_mean_abs_delta": float(endpoint_delta.mean().item()),
        "linear_half_width_normalized_l1_reference": float(
            (endpoint_delta.mean() / reference_half.mean().clamp_min(eps)).item()
        ),
        "linear_half_width_normalized_l1_actual": float(
            (endpoint_delta.mean() / original_half.mean().clamp_min(eps)).item()
        ),
        "cap_hit_fraction": (
            float((scaled > float(cap)).float().mean().item())
            if cap is not None
            else 0.0
        ),
        "far_clamp_fraction": float(
            ((changed[:, 0] + changed_width) >= MAX_LOG_DEPTH)
            .float()
            .mean()
            .item()
        ),
        "changed_log_half_width": _distribution(changed_width),
        "changed_linear_half_width": _distribution(changed_half),
    }


class DepthWidthCalibrationRecorder:
    """Join calibrated input evidence with compact phase/logit responses."""

    def __init__(
        self,
        *,
        num_patches: int,
        family: str,
        strategy: str,
        query_samples: int = 4,
        key_samples_per_camera: int = 4,
    ) -> None:
        self.num_patches = num_patches
        self.family = family
        self.strategy = strategy
        self.query_samples = query_samples
        self.key_samples_per_camera = key_samples_per_camera
        self.records: list[dict[str, object]] = []
        self._scene: Optional[dict[str, object]] = None
        self._forward_index: Optional[int] = None
        self._pending: Optional[dict[str, object]] = None
        self.capacity_manifest = build_capacity_manifest()

    def begin_scene(self, *, scene_id: str, scene_index: int, mc_index: int) -> None:
        if not scene_id or min(scene_index, mc_index) < 0:
            raise ValueError("calibration scene metadata is invalid")
        if self._pending is not None:
            raise RuntimeError("cannot change scene with a pending layer")
        self._scene = {
            "scene_id": scene_id,
            "scene_index": scene_index,
            "mc_index": mc_index,
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
    ) -> DepthObservabilityRequest:
        if self._scene is None or self._forward_index is None:
            raise RuntimeError("scene and target view must precede calibration")
        if self._pending is not None:
            raise RuntimeError("previous calibration layer was not recorded")
        scene_id = str(self._scene["scene_id"])
        references, counterfactuals, dose_cache = build_calibration_comparisons(
            actual_predicted_d,
            num_patches=self.num_patches,
            scene_id=scene_id,
            layer_index=layer_index,
        )
        B, N, _ = actual_predicted_d.shape
        C = N // self.num_patches
        width_cameras = []
        for batch in range(B):
            for camera in range(C):
                start = camera * self.num_patches
                stop = start + self.num_patches
                original = actual_predicted_d[batch, start:stop].detach().float().cpu()
                width_cameras.append(
                    {
                        "batch_index": batch,
                        "camera_index": camera,
                        "actual_log_half_width": _distribution(original[:, 1].abs()),
                        "comparisons": {
                            mode: _comparison_record(
                                original,
                                references[mode][batch, start:stop].detach().float().cpu(),
                                counterfactuals[mode][batch, start:stop]
                                .detach()
                                .float()
                                .cpu(),
                                ARM_SPECS[mode],
                            )
                            for mode in COUNTERFACTUAL_MODES
                        },
                    }
                )
        shadows: list[SegmentShadowProbe] = []
        if self.family == "segment_bounds":
            zero = dose_cache[(0.0, None)]
            low = dose_cache[(0.125, 0.5)]
            mid = dose_cache[(0.25, 1.0)]
            low_reverse = counterfactuals["route_low_reverse"]
            mid_reverse = counterfactuals["route_mid_reverse"]
            shadow_references = {
                "low_correct": zero,
                "low_reverse": low,
                "mid_correct": zero,
                "mid_reverse": mid,
            }
            shadow_counterfactuals = {
                "low_correct": low,
                "low_reverse": low_reverse,
                "mid_correct": mid,
                "mid_reverse": mid_reverse,
            }
            for name, bank in build_shadow_banks().items():
                shadows.append(
                    SegmentShadowProbe(
                        name=name,
                        omega=torch.from_numpy(bank.omega).to(actual_predicted_d),
                        head_groups=bank.head_groups,
                        comparison_references=shadow_references,
                        counterfactuals=shadow_counterfactuals,
                    )
                )
        routing_namespace = (
            f"depth-width-calibration-v1:{scene_id}:{layer_index}"
        ).encode("ascii")
        self._pending = {
            **self._scene,
            "target_view_index": self._forward_index,
            "layer_index": layer_index,
            "batch_size": B,
            "camera_count": C,
            "tokens_per_camera": self.num_patches,
            "original_equals_actual": original_predicted_d is actual_predicted_d,
            "routing_seed_sha256": hashlib.sha256(routing_namespace).hexdigest(),
            "width_cameras": width_cameras,
        }
        return DepthObservabilityRequest(
            layer_index=layer_index,
            original_predicted_d=original_predicted_d,
            actual_predicted_d=actual_predicted_d,
            counterfactuals=counterfactuals,
            comparison_references=references,
            shadow_segments=tuple(shadows),
            recorder=self,
            query_samples=self.query_samples,
            key_samples_per_camera=self.key_samples_per_camera,
            compact_phase=True,
        )

    def record_layer(self, payload: Mapping[str, object]) -> None:
        if self._pending is None:
            raise RuntimeError("calibration response has no pending input record")
        if payload.get("layer_index") != self._pending["layer_index"]:
            raise ValueError("calibration response layer does not match request")
        responses = payload.get("responses")
        shadows = payload.get("shadow_responses")
        expected_shadows = (
            {row["name"] for row in self.capacity_manifest["probes"]}
            if self.family == "segment_bounds"
            else set()
        )
        if not isinstance(responses, Mapping) or tuple(responses) != COUNTERFACTUAL_MODES:
            raise ValueError("calibration response modes are incomplete or reordered")
        if not isinstance(shadows, Mapping) or set(shadows) != expected_shadows:
            raise ValueError("calibration shadow responses are incomplete")
        self.records.append(
            {
                **self._pending,
                "generator_state_sha256": payload["generator_state_sha256"],
                "responses": dict(responses),
                "shadow_responses": dict(shadows),
            }
        )
        self._pending = None

    def finalize(self, path: Path, *, metadata: Mapping[str, object]) -> None:
        if self._pending is not None or not self.records:
            raise RuntimeError("cannot finalize incomplete calibration evidence")
        if not path.is_absolute():
            raise ValueError("calibration output path must be absolute")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "completed",
            "family": self.family,
            "strategy": self.strategy,
            "counterfactual_modes": list(COUNTERFACTUAL_MODES),
            "arm_specs": ARM_SPECS,
            "capacity_manifest_sha256": self.capacity_manifest["sha256"],
            "query_samples": self.query_samples,
            "key_samples_per_camera": self.key_samples_per_camera,
            "record_count": len(self.records),
            "layer_indices": sorted({row["layer_index"] for row in self.records}),
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
    "COUNTERFACTUAL_MODES",
    "DepthWidthCalibrationRecorder",
    "build_calibration_comparisons",
]
