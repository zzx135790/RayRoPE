"""Tests for per-scene metric records used by crossed bootstrap."""

import numpy as np
import pytest
import torch


def test_scene_metric_record_is_json_ready_and_uses_scene_directory():
    from nvs.metric_records import build_scene_metric_record

    paths = np.array([["/dataset/scene-42/images/frame-000.png"]])
    record = build_scene_metric_record(
        paths,
        psnr=torch.tensor(21.5),
        ssim=torch.tensor(0.81),
        lpips=torch.tensor(0.12),
    )

    assert record == {
        "scene_id": "scene-42",
        "psnr": 21.5,
        "ssim": pytest.approx(0.81),
        "lpips": pytest.approx(0.12),
    }


def test_merge_rank_records_sorts_and_rejects_duplicate_scene_ids():
    from nvs.metric_records import merge_rank_metric_records

    merged = merge_rank_metric_records(
        [
            [{"scene_id": "scene-b", "psnr": 2.0, "ssim": 0.2, "lpips": 0.8}],
            [{"scene_id": "scene-a", "psnr": 1.0, "ssim": 0.1, "lpips": 0.9}],
        ]
    )

    assert [record["scene_id"] for record in merged] == ["scene-a", "scene-b"]

    with pytest.raises(ValueError, match="duplicate scene_id"):
        merge_rank_metric_records(
            [
                [{"scene_id": "same", "psnr": 2.0, "ssim": 0.2, "lpips": 0.8}],
                [{"scene_id": "same", "psnr": 1.0, "ssim": 0.1, "lpips": 0.9}],
            ]
        )


def test_build_metrics_payload_includes_sorted_per_scene_records():
    from nvs.metric_records import build_metrics_payload

    payload = build_metrics_payload(
        label="",
        step=15000,
        n_total=2,
        psnr=20.0,
        ssim=0.8,
        lpips=0.12,
        records_by_rank=[
            [{"scene_id": "scene-b", "psnr": 21.0, "ssim": 0.9, "lpips": 0.1}],
            [{"scene_id": "scene-a", "psnr": 19.0, "ssim": 0.7, "lpips": 0.14}],
        ],
    )

    assert payload["psnr"] == 20.0
    assert payload["n_total"] == 2
    assert [row["scene_id"] for row in payload["per_scene"]] == [
        "scene-a",
        "scene-b",
    ]
