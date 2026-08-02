from __future__ import annotations

import pytest
import torch

from nvs.depth_width_scale_calibration import (
    DepthWidthScaleCalibrationRecorder,
    _target_frequency_banks,
    build_width_counterfactuals,
)
from tokenmap.experiments.flag_rope.depth_width_scale_calibration import (
    CALIBRATORS,
    SCALES,
    arm_name,
    width_name,
)


def test_raw_and_median_unit_widths_have_distinct_units() -> None:
    predicted = torch.tensor([[[0.0, -2.0], [0.0, 4.0]]])
    rows = build_width_counterfactuals(predicted, layer_median=2.0)

    torch.testing.assert_close(
        rows[width_name("raw_multiplier", 0.125)][..., 1],
        torch.tensor([[0.25, 0.5]]),
    )
    torch.testing.assert_close(
        rows[width_name("median_unit", 0.125)][..., 1],
        torch.tensor([[0.125, 0.25]]),
    )
    assert len(rows) == len(CALIBRATORS) * len(SCALES)


def test_target_frequency_banks_preserve_negative_control_coordinates() -> None:
    point_one, _ = _target_frequency_banks("ray_point", 1)
    point_four, _ = _target_frequency_banks("ray_point", 4)
    torch.testing.assert_close(point_four["point"], point_one["point"] / 4)

    segment_one, _ = _target_frequency_banks("segment_bounds", 1)
    segment_four, _ = _target_frequency_banks("segment_bounds", 4)
    torch.testing.assert_close(
        segment_four["segment_bounds"][..., :7],
        segment_one["segment_bounds"][..., :7],
    )
    torch.testing.assert_close(
        segment_four["segment_bounds"][..., 7],
        segment_one["segment_bounds"][..., 7] / 4,
    )


def _recorder(stage: str) -> DepthWidthScaleCalibrationRecorder:
    return DepthWidthScaleCalibrationRecorder(
        num_patches=2,
        family="ray_point",
        strategy="linearized_shared_sample",
        stage=stage,
        layer_medians={str(layer): 2.0 for layer in range(6)},
        query_samples=1,
        key_samples_per_camera=1,
    )


def test_source_stage_records_raw_widths_without_counterfactual_request() -> None:
    recorder = _recorder("source_audit")
    recorder.begin_scene(
        scene_id="scene-a", scene_index=0, mc_index=0, context_baseline_max=1.0
    )
    recorder.begin_forward(0)
    predicted = torch.tensor([[[0.0, -2.0], [0.0, 4.0]]]).repeat(1, 5, 1)

    request = recorder.request(
        layer_index=0,
        original_predicted_d=predicted,
        actual_predicted_d=predicted,
    )

    assert request is None
    assert recorder.records[0]["raw_log_half_width"] == [2.0, 4.0] * 5


def test_selection_stage_builds_18_widths_and_paired_wavelength_shadows() -> None:
    recorder = _recorder("selection_evaluation")
    recorder.begin_scene(
        scene_id="scene-a", scene_index=0, mc_index=0, context_baseline_max=1.0
    )
    recorder.begin_forward(0)
    predicted = torch.tensor([[[0.0, 2.0], [0.0, 4.0]]]).repeat(1, 5, 1)

    request = recorder.request(
        layer_index=0,
        original_predicted_d=predicted,
        actual_predicted_d=predicted,
    )

    assert request is not None
    assert len(request.counterfactuals) == 18
    assert set(request.counterfactuals) == {
        arm_name(calibrator, scale, 1)
        for calibrator in CALIBRATORS for scale in SCALES
    }
    assert [probe.name for probe in request.shadow_frequencies] == [
        "wavelength_x2", "wavelength_x4"
    ]
    assert all(len(probe.counterfactuals) == 18 for probe in request.shadow_frequencies)


def test_median_unit_rejects_missing_train_layer_median() -> None:
    recorder = DepthWidthScaleCalibrationRecorder(
        num_patches=2,
        family="ray_point",
        strategy="linearized_shared_sample",
        stage="selection_evaluation",
    )
    recorder.begin_scene(
        scene_id="scene-a", scene_index=0, mc_index=0, context_baseline_max=1.0
    )
    recorder.begin_forward(0)
    predicted = torch.ones(1, 10, 2)

    with pytest.raises(ValueError, match="finite positive"):
        recorder.request(
            layer_index=0,
            original_predicted_d=predicted,
            actual_predicted_d=predicted,
        )
