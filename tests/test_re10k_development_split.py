from __future__ import annotations

import json

import pytest

from nvs.re10k_dataset import RE10K_EvalDataset
from nvs.trainval import _re10k_train_scenes


def _scene(root, name):
    directory = root / name
    directory.mkdir()
    (directory / "transforms.json").write_text("{}", encoding="utf-8")
    return directory


def test_train_scene_manifest_preserves_exact_declared_order(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    _scene(train, "a")
    _scene(train, "b")
    _scene(train, "c")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "train_scenes": ["c", "a"],
        "selection_scenes": ["b"],
    }), encoding="utf-8")

    assert _re10k_train_scenes(str(train), str(manifest)) == [
        str(train / "c"), str(train / "a")
    ]


def test_train_scene_manifest_rejects_missing_or_duplicate_scenes(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    _scene(train, "a")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "train_scenes": ["a", "a"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _re10k_train_scenes(str(train), str(manifest))

    manifest.write_text(json.dumps({
        "schema_version": 1, "train_scenes": ["missing"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="unavailable"):
        _re10k_train_scenes(str(train), str(manifest))


def test_eval_dataset_accepts_absolute_development_index(tmp_path):
    eval_root = tmp_path / "train"
    eval_root.mkdir()
    scene = _scene(eval_root, "selection-scene")
    index = tmp_path / "selection-index.json"
    index.write_text(json.dumps({
        "selection-scene": {"context": [0, 1, 2, 3], "target": [4, 5, 6]},
    }), encoding="utf-8")

    dataset = RE10K_EvalDataset(
        folder=str(eval_root),
        input_views=4,
        supervise_views=3,
        test_index_fp=str(index.resolve()),
    )

    assert len(dataset) == 1
    assert dataset.data_dirs.tolist() == [str(scene).encode("utf-8")]
