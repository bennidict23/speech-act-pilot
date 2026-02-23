"""Tests for tasks.py — task scenario well-formedness."""
from __future__ import annotations

import pytest

from mock_tools import TASK_ERROR_STATES
from tasks import TASK_SCENARIOS, get_task_scenario

TASK_IDS = list(TASK_SCENARIOS.keys())


class TestTaskDefinitions:
    """All 5 task scenarios must be well-formed."""

    def test_exactly_5_tasks(self):
        assert len(TASK_SCENARIOS) == 5

    def test_task_ids_match_error_states(self):
        """TASK_SCENARIOS keys must exactly match TASK_ERROR_STATES keys."""
        assert set(TASK_SCENARIOS.keys()) == set(TASK_ERROR_STATES.keys())

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_required_fields_nonempty(self, task_id):
        task = TASK_SCENARIOS[task_id]
        assert task.task_id, f"{task_id}: empty task_id"
        assert task.goal, f"{task_id}: empty goal"
        assert len(task.tools) >= 2, f"{task_id}: needs at least 2 tools"
        assert task.error_tool, f"{task_id}: empty error_tool"

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_error_tool_in_tool_list(self, task_id):
        """The error_tool must be one of the available tools."""
        task = TASK_SCENARIOS[task_id]
        tool_names = {t.name for t in task.tools}
        assert task.error_tool in tool_names, (
            f"{task_id}: error_tool '{task.error_tool}' not in {tool_names}"
        )

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_task_id_field_matches_key(self, task_id):
        task = TASK_SCENARIOS[task_id]
        assert task.task_id == task_id

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_frozen(self, task_id):
        task = TASK_SCENARIOS[task_id]
        with pytest.raises(AttributeError):
            task.goal = "modified"  # type: ignore[misc]

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_tool_schemas_have_fields(self, task_id):
        task = TASK_SCENARIOS[task_id]
        for tool in task.tools:
            assert tool.name, f"{task_id}: tool missing name"
            assert tool.description, f"{task_id}/{tool.name}: missing description"
            assert tool.parameters is not None

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_tools_between_2_and_4(self, task_id):
        task = TASK_SCENARIOS[task_id]
        assert 2 <= len(task.tools) <= 4

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_goal_does_not_mention_error(self, task_id):
        """Goals should describe the task, not the error."""
        lower = TASK_SCENARIOS[task_id].goal.lower()
        assert "error" not in lower
        assert "fail" not in lower

    def test_get_task_scenario(self):
        task = get_task_scenario("file_write")
        assert task.task_id == "file_write"

    def test_get_task_scenario_unknown_raises(self):
        with pytest.raises(KeyError):
            get_task_scenario("nonexistent")
