from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import nvs.re10k_dataset as re10k_dataset
from nvs.re10k_dataset import RE10K_EvalDataset


ASSETS = Path(re10k_dataset.__file__).resolve().parent.parent / "assets"


def _construct_and_capture_index(
    monkeypatch: pytest.MonkeyPatch,
    folder: Path,
    **dataset_kwargs,
) -> tuple[RE10K_EvalDataset, Path]:
    real_open = builtins.open
    opened_json_files = []

    def tracking_open(file, *args, **kwargs):
        path = Path(file)
        if path.suffix == ".json":
            opened_json_files.append(path)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    dataset = RE10K_EvalDataset(folder=str(folder), **dataset_kwargs)
    assert len(opened_json_files) == 1
    return dataset, opened_json_files[0]


def test_eval_dataset_uses_absolute_test_index_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = tmp_path / "re10k-test"
    folder.mkdir()
    absolute_index = tmp_path / "launcher-selected-index.json"
    absolute_index.write_text("{}")

    dataset, opened_index = _construct_and_capture_index(
        monkeypatch,
        folder,
        test_index_fp=str(absolute_index),
    )

    assert opened_index.resolve() == absolute_index.resolve()
    assert len(dataset) == 0


def test_eval_dataset_keeps_relative_test_index_in_repository_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    folder = tmp_path / "re10k-test"
    unrelated_cwd = tmp_path / "unrelated-cwd"
    folder.mkdir()
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    dataset, opened_index = _construct_and_capture_index(
        monkeypatch,
        folder,
        test_index_fp="evaluation_index_re10k_4ctx.json",
    )

    assert opened_index.resolve() == (ASSETS / "evaluation_index_re10k_4ctx.json")
    assert len(dataset) == 0


@pytest.mark.parametrize(
    ("dataset_kwargs", "expected_name"),
    (
        ({}, "evaluation_index_re10k.json"),
        ({"input_views": 4}, "evaluation_index_re10k_4ctx.json"),
        ({"render_video": True}, "evaluation_index_re10k_video.json"),
    ),
)
def test_eval_dataset_none_index_keeps_existing_default_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_kwargs,
    expected_name: str,
) -> None:
    folder = tmp_path / "re10k-test"
    folder.mkdir()

    dataset, opened_index = _construct_and_capture_index(
        monkeypatch,
        folder,
        test_index_fp=None,
        **dataset_kwargs,
    )

    assert opened_index.resolve() == (ASSETS / expected_name)
    assert len(dataset) == 0
