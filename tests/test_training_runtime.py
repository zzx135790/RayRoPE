from __future__ import annotations

import inspect

import pytest

from pos_enc.utils.runner import Launcher, build_training_runtime_payload


def test_training_runtime_payload_reports_only_optimizer_updates() -> None:
    payload = build_training_runtime_payload(
        first_step=0,
        last_step=4999,
        wall_seconds=250.0,
    )

    assert payload == {
        "schema_version": 1,
        "kind": "training-update-runtime",
        "first_step": 0,
        "last_step": 4999,
        "training_updates": 5000,
        "training_wall_seconds": 250.0,
        "training_seconds_per_update": 0.05,
        "timing_boundary": "before_first_training_iteration_to_after_final_optimizer_update",
        "cuda_synchronized_at_boundaries": True,
    }
    with pytest.raises(ValueError, match="last_step"):
        build_training_runtime_payload(first_step=3, last_step=2, wall_seconds=1.0)


def test_training_timer_finishes_before_checkpoint_and_evaluation() -> None:
    source = inspect.getsource(Launcher.train)

    finish = source.index("self._finish_training_runtime")
    assert finish < source.index("self.save_checkpoint")
    assert finish < source.index("self.test_iteration")
