from __future__ import annotations

import importlib
import sys

import pytest


def test_re10k_selection_does_not_require_co3d_or_objaverse_environment(monkeypatch) -> None:
    for key in ("OBJV_DIR", "CO3D_DIR", "CO3D_ANNOTATION_DIR", "CO3D_DEPTH_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RE10K_TRAIN_DIR", "/synthetic/re10k/train")
    monkeypatch.setenv("RE10K_TEST_DIR", "/synthetic/re10k/test")
    sys.modules.pop("nvs.trainval", None)

    module = importlib.import_module("nvs.trainval")

    assert module.dataset_paths_for("re10k").train == "/synthetic/re10k/train"


def test_co3d_selection_reports_only_the_missing_co3d_variables(monkeypatch) -> None:
    from nvs.runtime_paths import DatasetEnvironmentError, dataset_paths_for

    for key in ("CO3D_DIR", "CO3D_ANNOTATION_DIR", "CO3D_DEPTH_DIR"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(DatasetEnvironmentError) as raised:
        dataset_paths_for("co3d")

    assert raised.value.missing == ("CO3D_ANNOTATION_DIR", "CO3D_DEPTH_DIR", "CO3D_DIR")


def test_selected_dataset_reads_only_its_declared_environment() -> None:
    from nvs.runtime_paths import dataset_paths_for

    objaverse = dataset_paths_for("objaverse", {"OBJV_DIR": "/synthetic/objaverse"})
    co3d = dataset_paths_for(
        "co3d",
        {
            "CO3D_DIR": "/synthetic/co3d/images",
            "CO3D_ANNOTATION_DIR": "/synthetic/co3d/annotations",
            "CO3D_DEPTH_DIR": "/synthetic/co3d/depth",
        },
    )

    assert objaverse.root == "/synthetic/objaverse"
    assert co3d.root == "/synthetic/co3d/images"
    assert co3d.annotation == "/synthetic/co3d/annotations"
    assert co3d.depth == "/synthetic/co3d/depth"


def test_selected_dataset_rejects_relative_paths() -> None:
    from nvs.runtime_paths import DatasetEnvironmentError, dataset_paths_for

    with pytest.raises(
        DatasetEnvironmentError, match=r"RE10K_TRAIN_DIR.*relative/train"
    ):
        dataset_paths_for(
            "re10k",
            {
                "RE10K_TRAIN_DIR": "relative/train",
                "RE10K_TEST_DIR": "/synthetic/re10k/test",
            },
        )
