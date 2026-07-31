from __future__ import annotations

import json

import torch

from nvs.depth_width_observability import (
    COUNTERFACTUAL_MODES,
    DepthWidthObservabilityRecorder,
    _rank_correlation,
    depth_counterfactuals,
)
from nvs.lvsm import _transform_depth_uncertainty


def test_counterfactuals_preserve_centres_and_match_current_roll() -> None:
    predicted = torch.tensor(
        [[
            [1.0, 0.1],
            [2.0, -0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 1.1],
            [6.0, 1.2],
            [7.0, 1.3],
            [8.0, 1.4],
        ]]
    )
    values = depth_counterfactuals(predicted, num_patches=4)

    assert tuple(values) == COUNTERFACTUAL_MODES
    for value in values.values():
        torch.testing.assert_close(value[..., 0], predicted[..., 0])
    torch.testing.assert_close(values["zero"][..., 1], torch.zeros(1, 8))
    torch.testing.assert_close(values["half"][..., 1], predicted[..., 1] * 0.5)
    torch.testing.assert_close(values["double"][..., 1], predicted[..., 1] * 2.0)
    torch.testing.assert_close(
        values["permute_within_camera"],
        _transform_depth_uncertainty(
            predicted, "permute_within_camera", num_patches=4
        ),
    )


def test_permutation_rank_correlation_uses_average_ranks_for_ties() -> None:
    left = torch.tensor([1.0, 1.0, 2.0, 2.0])
    right = torch.tensor([1.0, 2.0, 1.0, 2.0])

    assert _rank_correlation(left, right) == 0.0


def test_recorder_joins_width_and_response_evidence(tmp_path) -> None:
    predicted = torch.tensor(
        [[
            [0.0, 0.1],
            [0.1, 0.2],
            [0.2, 0.3],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.6],
            [0.6, 0.7],
            [0.7, 0.8],
        ]]
    )
    recorder = DepthWidthObservabilityRecorder(
        num_patches=4,
        family="ray_point",
        strategy="linearized_shared_sample",
        query_samples=2,
        key_samples_per_camera=2,
    )
    recorder.begin_scene(scene_id="scene-a", scene_index=0, mc_index=0)
    recorder.begin_forward(0)
    request = recorder.request(
        layer_index=2,
        original_predicted_d=predicted,
        actual_predicted_d=predicted,
    )
    recorder.record_layer(
        {
            "layer_index": 2,
            "responses": {
                mode: {"phase": {"heads": [], "pairs": []}, "attention_logits": []}
                for mode in COUNTERFACTUAL_MODES
            },
        }
    )
    output = (tmp_path / "audit.json").resolve()
    recorder.finalize(output, metadata={"step": 4999})

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert request.layer_index == 2
    assert payload["status"] == "completed"
    assert payload["record_count"] == 1
    record = payload["records"][0]
    assert record["scene_id"] == "scene-a"
    assert len(record["width_cameras"]) == 2
    intervention = record["width_cameras"][0]["interventions"][
        "permute_within_camera"
    ]
    assert intervention["mean_abs_log_width_delta"] > 0.0
    assert intervention["position_rank_correlation"] < 1.0
