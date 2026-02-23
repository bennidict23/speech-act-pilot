"""Tests for run_experiment.py — seed computation, JSONL serialization,
error handling (M1), loop count, and resume logic.

NO live LLM required.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent import AgentConfig, AgentStep, AgentTrajectory
from run_experiment import (
    compute_loop_count,
    compute_seed,
    load_completed_runs,
    make_error_record,
    trajectory_to_record,
)


# ---------------------------------------------------------------------------
# compute_seed
# ---------------------------------------------------------------------------


class TestComputeSeed:

    def test_formula(self):
        assert compute_seed(0, 0, 0) == 0
        assert compute_seed(1, 2, 3) == 10203
        assert compute_seed(4, 4, 19) == 40419

    def test_no_overlap(self):
        """All 500 seeds must be unique."""
        seeds = set()
        for ti in range(5):
            for si in range(5):
                for run in range(20):
                    seeds.add(compute_seed(ti, si, run))
        assert len(seeds) == 500

    def test_non_negative(self):
        for ti in range(5):
            for si in range(5):
                for run in range(20):
                    assert compute_seed(ti, si, run) >= 0


# ---------------------------------------------------------------------------
# compute_loop_count
# ---------------------------------------------------------------------------


class TestComputeLoopCount:

    def test_no_retries(self):
        assert compute_loop_count(["correct_recovery"]) == 0

    def test_single_retry(self):
        assert compute_loop_count(["retry", "correct_recovery"]) == 1

    def test_consecutive_retries(self):
        assert compute_loop_count(["retry", "retry", "retry", "correct_recovery"]) == 3

    def test_non_consecutive(self):
        assert compute_loop_count(["retry", "unknown", "retry", "retry"]) == 2

    def test_empty(self):
        assert compute_loop_count([]) == 0

    def test_all_retries(self):
        assert compute_loop_count(["retry"] * 10) == 10


# ---------------------------------------------------------------------------
# trajectory_to_record
# ---------------------------------------------------------------------------


def _make_trajectory(
    outcome: str = "success",
    action_types: list[str] | None = None,
) -> AgentTrajectory:
    """Create a minimal synthetic AgentTrajectory."""
    if action_types is None:
        action_types = ["retry", "correct_recovery"]
    steps = tuple(
        AgentStep(
            step_number=i + 1,
            raw_output=f"raw_{i}",
            thought=f"thought_{i}",
            action=f"action_{i}",
            observation=f"obs_{i}",
            action_type=at,
        )
        for i, at in enumerate(action_types)
    )
    config = AgentConfig(
        model="test",
        base_url="http://localhost:8001/v1",
        temperature=0.7,
        max_steps=10,
        seed=42,
    )
    return AgentTrajectory(
        task_id="file_write",
        style="neutral",
        phase="b1_minimal",
        config=config,
        steps=steps,
        outcome=outcome,
        total_steps=len(steps),
    )


class TestTrajectoryToRecord:

    def test_required_keys(self):
        traj = _make_trajectory()
        record = trajectory_to_record("file_write", "neutral", 0, 42, traj)
        required = {
            "task", "style", "run", "seed", "recovered", "steps",
            "loop_count", "actions", "action_types", "outcome",
            "trajectory", "error",
        }
        assert required.issubset(record.keys())

    def test_recovered_true_on_success(self):
        traj = _make_trajectory(outcome="success")
        record = trajectory_to_record("file_write", "neutral", 0, 42, traj)
        assert record["recovered"] is True

    def test_recovered_false_on_failure(self):
        traj = _make_trajectory(outcome="max_steps")
        record = trajectory_to_record("file_write", "neutral", 0, 42, traj)
        assert record["recovered"] is False

    def test_action_types_extracted(self):
        traj = _make_trajectory(action_types=["retry", "retry", "correct_recovery"])
        record = trajectory_to_record("file_write", "neutral", 0, 42, traj)
        assert record["action_types"] == ["retry", "retry", "correct_recovery"]
        assert record["loop_count"] == 2

    def test_error_is_none(self):
        traj = _make_trajectory()
        record = trajectory_to_record("file_write", "neutral", 0, 42, traj)
        assert record["error"] is None

    def test_serializable_to_json(self):
        traj = _make_trajectory()
        record = trajectory_to_record("file_write", "neutral", 0, 42, traj)
        serialized = json.dumps(record)
        assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# make_error_record (M1)
# ---------------------------------------------------------------------------


class TestMakeErrorRecord:

    def test_required_keys(self):
        record = make_error_record("file_write", "neutral", 0, 42, "boom")
        required = {
            "task", "style", "run", "seed", "recovered", "steps",
            "loop_count", "actions", "action_types", "outcome",
            "trajectory", "error",
        }
        assert required.issubset(record.keys())

    def test_recovered_false(self):
        record = make_error_record("file_write", "neutral", 0, 42, "boom")
        assert record["recovered"] is False

    def test_error_message(self):
        record = make_error_record("file_write", "neutral", 0, 42, "timeout")
        assert record["error"] == "timeout"

    def test_zero_steps(self):
        record = make_error_record("file_write", "neutral", 0, 42, "err")
        assert record["steps"] == 0
        assert record["actions"] == []


# ---------------------------------------------------------------------------
# load_completed_runs (resume)
# ---------------------------------------------------------------------------


class TestLoadCompletedRuns:

    def test_nonexistent_file(self, tmp_path):
        result = load_completed_runs(tmp_path / "missing.jsonl")
        assert result == set()

    def test_loads_completed(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text(
            json.dumps({"task": "file_write", "style": "neutral", "run": 0}) + "\n"
            + json.dumps({"task": "api_call", "style": "directive", "run": 5}) + "\n"
        )
        result = load_completed_runs(f)
        assert ("file_write", "neutral", 0) in result
        assert ("api_call", "directive", 5) in result
        assert len(result) == 2

    def test_skips_malformed(self, tmp_path):
        f = tmp_path / "out.jsonl"
        f.write_text("not json\n" + json.dumps({"task": "x", "style": "y", "run": 0}) + "\n")
        result = load_completed_runs(f)
        assert len(result) == 1
