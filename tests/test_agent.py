"""Tests for agent.py — ReAct parsing and classify_action contract.

These tests do NOT require a running LLM server. They test pure
parsing functions and prove why the CC2 contract matters.
"""
from __future__ import annotations

import pytest

from agent import build_system_prompt, parse_react_output
from mock_tools import ActionType, classify_action
from tasks import TASK_SCENARIOS


# ---------------------------------------------------------------------------
# parse_react_output
# ---------------------------------------------------------------------------


class TestParseReactOutput:
    """parse_react_output must cleanly separate Thought from Action."""

    def test_standard_format(self):
        raw = (
            "Thought: I should try a different path\n"
            "Action: write_file(/tmp/reports/Q1.txt)"
        )
        thought, action = parse_react_output(raw)
        assert "different path" in thought
        assert "write_file" in action
        assert "Thought:" not in action
        assert "I should" not in action

    def test_action_only(self):
        raw = "Action: http_get(/api/v2/users/123)"
        thought, action = parse_react_output(raw)
        assert thought == ""
        assert "http_get" in action

    def test_no_action_marker(self):
        raw = "I'm thinking about what to do next"
        thought, action = parse_react_output(raw)
        assert thought == raw
        assert action == ""

    def test_multiline_thought(self):
        raw = (
            "Thought: The file write failed with permission denied.\n"
            "I need to try a different location.\n"
            "Action: write_file(/tmp/reports/Q1.txt)"
        )
        thought, action = parse_react_output(raw)
        assert "permission denied" in thought
        assert "different location" in thought
        assert "write_file" in action
        # Thought text must NOT leak into action
        assert "permission denied" not in action
        assert "different location" not in action

    def test_hallucinated_observation_truncated(self):
        raw = (
            "Thought: Let me try again\n"
            "Action: sql_query(SELECT * FROM sales)\n"
            "Observation: This should be ignored"
        )
        thought, action = parse_react_output(raw)
        assert "sql_query" in action
        assert "Observation" not in action
        assert "should be ignored" not in action

    def test_case_insensitive_markers(self):
        raw = "thought: thinking\naction: do_thing()"
        thought, action = parse_react_output(raw)
        assert "thinking" in thought
        assert "do_thing" in action

    def test_no_thought_prefix(self):
        """Text before Action: without explicit Thought: prefix."""
        raw = "Let me consider...\nAction: search('atlas')"
        thought, action = parse_react_output(raw)
        assert "consider" in thought
        assert "search" in action

    def test_empty_input(self):
        thought, action = parse_react_output("")
        assert thought == ""
        assert action == ""

    def test_whitespace_only(self):
        thought, action = parse_react_output("   \n  ")
        assert thought == ""
        assert action == ""

    def test_whitespace_handling(self):
        raw = "  Thought:  spaced out  \n  Action:  tool_call()  "
        thought, action = parse_react_output(raw)
        assert "spaced out" in thought
        assert "tool_call" in action

    def test_action_never_contains_thought_prefix(self):
        raw = "Thought: reasoning\nAction: tool()"
        _, action = parse_react_output(raw)
        assert not action.lower().startswith("thought")


# ---------------------------------------------------------------------------
# classify_action contract (CC2 CRITICAL)
# ---------------------------------------------------------------------------


class TestClassifyActionContract:
    """CRITICAL: classify_action must receive ONLY Action text.

    These tests prove that thought contamination causes misclassification.
    """

    def test_thought_retry_contaminates(self):
        """'retry' in thought must not affect action classification."""
        raw = (
            "Thought: I could retry but that won't work. "
            "Let me try a different path.\n"
            "Action: write_file(/tmp/reports/Q1.txt)"
        )
        thought, action = parse_react_output(raw)

        # Correct: classify ONLY the action
        result = classify_action(action, "file_write")
        assert result == ActionType.CORRECT_RECOVERY

    def test_thought_give_up_contaminates(self):
        """'impossible' in thought would trigger GIVE_UP on non-recovery action."""
        thought_text = "This seems impossible but let me try something"
        action_text = "http_get(/api/v1/users/999)"  # not a recovery action

        # If we wrongly classified combined text:
        combined = thought_text + "\n" + action_text
        contaminated = classify_action(combined, "api_call")
        assert contaminated == ActionType.GIVE_UP, (
            "Demonstrates contamination: 'impossible' in thought → GIVE_UP"
        )

        # Correct: classify only action
        clean = classify_action(action_text, "api_call")
        assert clean != ActionType.GIVE_UP

    def test_end_to_end_pipeline(self):
        """Parse output → classify ONLY action → correct result."""
        raw = (
            "Thought: I cannot retry this. The path is impossible. "
            "Let me use a different directory.\n"
            "Action: write_file(/tmp/reports/Q1.txt, content='Q1 report')"
        )
        thought, action = parse_react_output(raw)

        # Thought has misclassification triggers
        assert "retry" in thought.lower()
        assert "impossible" in thought.lower()

        # But action alone is correctly classified
        result = classify_action(action, "file_write")
        assert result == ActionType.CORRECT_RECOVERY

    def test_broaden_in_thought_does_not_leak(self):
        """'broaden' in thought should not matter if action is different."""
        raw = (
            "Thought: Maybe I should broaden the search.\n"
            "Action: search('Project Atlas exact match')"
        )
        thought, action = parse_react_output(raw)

        # 'broaden' is a recovery keyword for search task
        assert "broaden" in thought.lower()
        # But action doesn't contain it
        result = classify_action(action, "search")
        assert result != ActionType.CORRECT_RECOVERY


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """System prompt must be neutral and well-formed."""

    @pytest.mark.parametrize("task_id", list(TASK_SCENARIOS.keys()))
    def test_contains_goal(self, task_id):
        task = TASK_SCENARIOS[task_id]
        prompt = build_system_prompt(task)
        assert task.goal in prompt

    @pytest.mark.parametrize("task_id", list(TASK_SCENARIOS.keys()))
    def test_contains_all_tools(self, task_id):
        task = TASK_SCENARIOS[task_id]
        prompt = build_system_prompt(task)
        for tool in task.tools:
            assert tool.name in prompt

    @pytest.mark.parametrize("task_id", list(TASK_SCENARIOS.keys()))
    def test_no_experiment_leakage(self, task_id):
        """System prompt must not reveal the experiment."""
        task = TASK_SCENARIOS[task_id]
        prompt = build_system_prompt(task).lower()
        for keyword in ["experiment", "speech act", "pilot", "test", "study"]:
            assert keyword not in prompt, (
                f"System prompt leaks '{keyword}' for {task_id}"
            )

    def test_defines_react_format(self):
        task = TASK_SCENARIOS["file_write"]
        prompt = build_system_prompt(task)
        assert "Thought:" in prompt
        assert "Action:" in prompt
