from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
FLAG_SOURCE = WORKSPACE_ROOT / "worktrees/flag-rope/mainline/src"
sys.path.insert(0, str(FLAG_SOURCE))

from pos_enc.utils.runner import Launcher, LauncherConfig
from tokenmap.experiments.flag_rope import wandb_logging


class _FakeLogger:
    enabled = True
    run_id = "workspace-run-prope"
    run_url = "https://wandb.example/workspace-run-prope"
    entity = "entity"
    project = "tokenmap-flag-rope"

    def finish(self):
        pass


def test_required_wandb_writes_local_run_receipt(tmp_path, monkeypatch):
    captured = {}

    def init_logger(**kwargs):
        captured.update(kwargs)
        return _FakeLogger()

    monkeypatch.setattr(wandb_logging, "init_logger", init_logger)
    config = LauncherConfig(
        output_dir=str(tmp_path),
        wandb_enabled=True,
        wandb_mode="online",
        wandb_project="tokenmap-flag-rope",
        wandb_group="re10k-pose-noise",
        wandb_name="pose_prope",
        wandb_id="workspace-run-prope",
        wandb_resume="must",
        wandb_required=True,
    )

    launcher = Launcher(config)
    launcher.writer.close()

    assert captured["run_id"] == "workspace-run-prope"
    assert captured["resume"] == "must"
    assert captured["required"] is True
    assert json.loads((tmp_path / "wandb_run.json").read_text(encoding="utf-8")) == {
        "entity": "entity",
        "group": "re10k-pose-noise",
        "id": "workspace-run-prope",
        "mode": "online",
        "name": "pose_prope",
        "project": "tokenmap-flag-rope",
        "resume": "must",
        "schema_version": 1,
        "url": "https://wandb.example/workspace-run-prope",
    }


def test_required_wandb_failure_stops_before_training(tmp_path, monkeypatch):
    def init_logger(**kwargs):
        raise RuntimeError("authentication failed")

    monkeypatch.setattr(wandb_logging, "init_logger", init_logger)
    config = LauncherConfig(
        output_dir=str(tmp_path),
        wandb_enabled=True,
        wandb_required=True,
        wandb_id="workspace-run-prope",
    )

    with pytest.raises(RuntimeError, match="required W&B setup failed"):
        Launcher(config)


def test_run_always_closes_writer():
    class _Writer:
        closed = False

        def close(self):
            self.closed = True

    launcher = object.__new__(Launcher)
    launcher.config = LauncherConfig(test_only=False)
    launcher.world_rank = 0
    launcher.writer = _Writer()
    launcher.train = lambda: None

    launcher.run()

    assert launcher.writer.closed is True
