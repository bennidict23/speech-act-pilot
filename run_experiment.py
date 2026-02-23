"""Experiment runner for B1 phase: 5 tasks x 5 styles x 20 runs = 500 episodes.

Calls run_agent() for each condition, logs results to JSONL.
Each run_agent() call is wrapped in try/except (M1) — a single LLM
failure must not crash the experiment.
Supports resume: skips already-completed (task, style, run) triples.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from agent import AgentTrajectory, run_agent
from mock_tools import TASK_ERROR_STATES
from renderer import Phase, SpeechStyle

logger = logging.getLogger(__name__)

TASK_IDS: tuple[str, ...] = tuple(TASK_ERROR_STATES.keys())
STYLES: tuple[SpeechStyle, ...] = tuple(SpeechStyle)
RUNS_PER_CONDITION: int = 20


# ---------------------------------------------------------------------------
# Pure helpers (testable without LLM)
# ---------------------------------------------------------------------------

def compute_seed(task_index: int, style_index: int, run: int) -> int:
    """Deterministic seed from condition indices. Non-overlapping ranges."""
    return task_index * 10000 + style_index * 100 + run


def compute_loop_count(action_types: list[str]) -> int:
    """Length of the longest consecutive run of 'retry'."""
    max_run = 0
    current_run = 0
    for at in action_types:
        if at == "retry":
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def trajectory_to_record(
    task_id: str,
    style: str,
    run: int,
    seed: int,
    trajectory: AgentTrajectory,
) -> dict:
    """Convert an AgentTrajectory into the JSONL record dict."""
    actions = [step.action for step in trajectory.steps]
    action_types = [step.action_type for step in trajectory.steps]
    return {
        "task": task_id,
        "style": style,
        "run": run,
        "seed": seed,
        "recovered": trajectory.outcome == "success",
        "steps": trajectory.total_steps,
        "loop_count": compute_loop_count(action_types),
        "actions": actions,
        "action_types": action_types,
        "outcome": trajectory.outcome,
        "trajectory": asdict(trajectory),
        "error": None,
    }


def make_error_record(
    task_id: str,
    style: str,
    run: int,
    seed: int,
    error_msg: str,
) -> dict:
    """Create a JSONL record for a failed run (M1)."""
    return {
        "task": task_id,
        "style": style,
        "run": run,
        "seed": seed,
        "recovered": False,
        "steps": 0,
        "loop_count": 0,
        "actions": [],
        "action_types": [],
        "outcome": "error",
        "trajectory": None,
        "error": error_msg,
    }


def load_completed_runs(path: Path) -> set[tuple[str, str, int]]:
    """Read existing JSONL and return set of (task, style, run) completed."""
    completed: set[tuple[str, str, int]] = set()
    if not path.exists():
        return completed
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                completed.add((record["task"], record["style"], record["run"]))
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping malformed line %d: %s", line_num, exc)
    return completed


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_b1_experiment(
    output_path: Path,
    *,
    base_url: str = "http://localhost:8001/v1",
    model: str = "Qwen2.5-7B-Instruct",
    temperature: float = 0.7,
    max_steps: int = 10,
    runs_per_condition: int = RUNS_PER_CONDITION,
    task_filter: str | None = None,
    style_filter: str | None = None,
    dry_run: bool = False,
) -> None:
    """Run the B1 experiment: tasks x styles x runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed = load_completed_runs(output_path)
    if completed:
        logger.info("Resuming: %d runs already completed", len(completed))

    # Build condition list
    tasks = [t for t in TASK_IDS if task_filter is None or t == task_filter]
    styles = [s for s in STYLES if style_filter is None or s.value == style_filter]
    total = len(tasks) * len(styles) * runs_per_condition
    done = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    with open(output_path, "a") as f:
        for task_index, task_id in enumerate(TASK_IDS):
            if task_id not in tasks:
                continue
            for style_index, style in enumerate(STYLES):
                if style not in styles:
                    continue
                for run in range(runs_per_condition):
                    # Resume check
                    if (task_id, style.value, run) in completed:
                        skipped += 1
                        continue

                    seed = compute_seed(task_index, style_index, run)
                    done += 1
                    logger.info(
                        "[%d/%d] task=%s style=%s run=%d seed=%d",
                        done, total - skipped, task_id, style.value, run, seed,
                    )

                    if dry_run:
                        continue

                    # M1: try/except around run_agent — never crash the loop
                    try:
                        trajectory = run_agent(
                            task_id=task_id,
                            style=style,
                            phase=Phase.B1,
                            seed=seed,
                            base_url=base_url,
                            model=model,
                            temperature=temperature,
                            max_steps=max_steps,
                        )
                        record = trajectory_to_record(
                            task_id, style.value, run, seed, trajectory,
                        )
                    except Exception:
                        error_msg = traceback.format_exc()
                        logger.error(
                            "run_agent FAILED for task=%s style=%s run=%d:\n%s",
                            task_id, style.value, run, error_msg,
                        )
                        record = make_error_record(
                            task_id, style.value, run, seed, error_msg,
                        )
                        errors += 1

                    f.write(json.dumps(record) + "\n")
                    f.flush()

    elapsed = time.time() - start_time
    logger.info(
        "Done: %d runs completed, %d skipped (resume), %d errors, %.1fs elapsed",
        done, skipped, errors, elapsed,
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run B1 speech-act pilot experiment",
    )
    parser.add_argument(
        "--output", default="results/b1_results.jsonl",
        help="Output JSONL path (default: results/b1_results.jsonl)",
    )
    parser.add_argument("--base-url", default="http://localhost:8001/v1")
    parser.add_argument("--model", default="Qwen2.5-7B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--runs", type=int, default=RUNS_PER_CONDITION)
    parser.add_argument("--task", default=None, help="Run only this task")
    parser.add_argument("--style", default=None, help="Run only this style")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log conditions without calling LLM",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    run_b1_experiment(
        output_path=Path(args.output),
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_steps=args.max_steps,
        runs_per_condition=args.runs,
        task_filter=args.task,
        style_filter=args.style,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
