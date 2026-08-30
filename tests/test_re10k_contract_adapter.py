from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from nvs.re10k_contract import ContractRE10KEvalDataset


def _fixture(root: Path) -> tuple[Path, Path]:
    scene = root / "test" / "scene-a"
    images = scene / "images"
    images.mkdir(parents=True)
    frames = []
    for index in range(5):
        path = images / f"{index:05d}.png"
        Image.new("RGB", (64, 48), color=(index, index + 1, index + 2)).save(path)
        frames.append({
            "file_path": f"images/{index:05d}.png",
            "transform_matrix": [[1, 0, 0, index], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        })
    (scene / "transforms.json").write_text(json.dumps({
        "w": 64, "h": 48, "fl_x": 40, "fl_y": 41, "cx": 32, "cy": 24, "frames": frames,
    }), encoding="utf-8")
    index = root / "index.json"
    index.write_text(json.dumps({"scene-a": {"context": [0, 4], "target": [1, 2, 3]}}), encoding="utf-8")
    return scene, index


def _environment(monkeypatch, root: Path, mode: str, artifact: Path):
    values = {
        "WORKSPACE_DATASET_ID": "re10k",
        "WORKSPACE_DATASET_LOGICAL_ID": "re10k",
        "WORKSPACE_DATASET_ROOT": str(root),
        "WORKSPACE_DATASET_PROVIDER_ID": "re10k-transforms-v1",
        "WORKSPACE_DATASET_CONTRACT_VERSION": "1.0.0",
        "WORKSPACE_DATASET_PROFILE": "multiview-nvs-v1",
        "WORKSPACE_DATASET_CONSUMER_ADAPTER": "rayrope-multiview-nvs-v1",
        "WORKSPACE_DATASET_MODE": mode,
        "WORKSPACE_ARTIFACT_DIR": str(artifact),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_contract_eval_matches_legacy_fixed_views(monkeypatch, tmp_path):
    root = tmp_path / "re10k"
    scene, index = _fixture(root)
    _environment(monkeypatch, root, "contract", tmp_path / "artifacts")
    contract = ContractRE10KEvalDataset(
        str(root / "test"), patch_size=32, input_views=2, supervise_views=3,
        test_index_fp=str(index),
    )[0]

    from nvs.re10k_dataset import RE10K_EvalDataset

    legacy = RE10K_EvalDataset(
        str(root / "test"), patch_size=32, input_views=2, supervise_views=3,
        test_index_fp=str(index),
    )[0]
    assert torch.equal(contract["image"], legacy["image"])
    assert torch.equal(contract["K"], legacy["K"])
    assert torch.equal(contract["camtoworld"], legacy["camtoworld"])
    assert contract["image_path"] == legacy["image_path"]


def test_compare_eval_returns_legacy_and_publishes_receipt(monkeypatch, tmp_path):
    root = tmp_path / "re10k"
    _, index = _fixture(root)
    artifact = tmp_path / "artifacts"
    _environment(monkeypatch, root, "compare", artifact)
    item = ContractRE10KEvalDataset(
        str(root / "test"), patch_size=32, zoom_factor=1.2, random_zoom=True,
        input_views=2, supervise_views=3,
        test_index_fp=str(index),
    )[0]
    assert item["image"].shape == (5, 32, 32, 3)
    receipts = list((artifact / "dataset-contract-receipts" / "rayrope-multiview-nvs-v1").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["status"] == "verified"


def test_trainval_routes_declared_rayrope_contract_consumer(monkeypatch):
    monkeypatch.setenv("WORKSPACE_DATASET_MODE", "contract")
    monkeypatch.setenv(
        "WORKSPACE_DATASET_CONSUMER_ADAPTER", "rayrope-multiview-nvs-v1"
    )
    from nvs.trainval import _re10k_dataset_types

    train_type, eval_type = _re10k_dataset_types()
    assert train_type.__module__ == "nvs.re10k_contract"
    assert eval_type.__module__ == "nvs.re10k_contract"
