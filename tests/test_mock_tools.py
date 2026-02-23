"""Tests for mock_tools.py — determinism, state machine, action classification."""
from __future__ import annotations

import pytest

from mock_tools import (
    TASK_ERROR_STATES,
    ActionType,
    ToolStatus,
    classify_action,
    create_tool_session,
)


class TestErrorStates:
    """All 5 error states are well-formed and immutable."""

    def test_all_tasks_have_error_states(self):
        assert len(TASK_ERROR_STATES) == 5
        expected = {"file_write", "api_call", "db_query", "search", "code_exec"}
        assert set(TASK_ERROR_STATES.keys()) == expected

    def test_error_states_are_frozen(self):
        for state in TASK_ERROR_STATES.values():
            with pytest.raises(AttributeError):
                state.error_code = "changed"  # type: ignore[misc]

    def test_all_fields_nonempty(self):
        for task_id, state in TASK_ERROR_STATES.items():
            assert state.error_code, f"{task_id}: empty error_code"
            assert state.error_type, f"{task_id}: empty error_type"
            assert state.resource, f"{task_id}: empty resource"
            assert state.correct_recovery, f"{task_id}: empty correct_recovery"
            assert state.details, f"{task_id}: empty details"

    def test_no_retryable_tasks(self):
        """All 5 pilot tasks are non-retryable by design."""
        for task_id, state in TASK_ERROR_STATES.items():
            assert state.retryable is False, f"{task_id} should not be retryable"


class TestActionClassifier:
    """Rule-based action classification."""

    @pytest.mark.parametrize(
        "task_id,action,expected",
        [
            ("file_write", "I'll write to /tmp/reports/ instead", ActionType.CORRECT_RECOVERY),
            ("file_write", "Use a different directory for output", ActionType.CORRECT_RECOVERY),
            ("api_call", "Let me try /api/v2/users/123", ActionType.CORRECT_RECOVERY),
            ("api_call", "Switch to v2 endpoint", ActionType.CORRECT_RECOVERY),
            ("db_query", "First, let me list_tables to see what's available", ActionType.CORRECT_RECOVERY),
            ("db_query", "I should show tables first", ActionType.CORRECT_RECOVERY),
            ("search", "Let me broaden the search to just 'Atlas'", ActionType.CORRECT_RECOVERY),
            ("search", 'Search for "atlas" instead', ActionType.CORRECT_RECOVERY),
            ("code_exec", "I need to install pandas first", ActionType.CORRECT_RECOVERY),
            ("code_exec", "Run pip install pandas", ActionType.CORRECT_RECOVERY),
            ("code_exec", "install_package(pandas)", ActionType.CORRECT_RECOVERY),
            ("code_exec", "install_package('pandas')", ActionType.CORRECT_RECOVERY),
            ("code_exec", 'install_package("pandas")', ActionType.CORRECT_RECOVERY),
        ],
    )
    def test_correct_recovery_detected(self, task_id, action, expected):
        assert classify_action(action, task_id) == expected

    @pytest.mark.parametrize(
        "action,expected",
        [
            ("Let me retry the same action", ActionType.RETRY),
            ("Try again with the same parameters", ActionType.RETRY),
            ("I give up, this is impossible", ActionType.GIVE_UP),
            ("This task is impossible, I abort", ActionType.GIVE_UP),
            ("I need to ask user for help", ActionType.ASK_USER),
            ("Let me switch to a different tool", ActionType.SWITCH_TOOL),
        ],
    )
    def test_generic_actions(self, action, expected):
        assert classify_action(action, "file_write") == expected

    def test_unknown_action(self):
        result = classify_action("hmm let me think about this", "file_write")
        assert result == ActionType.UNKNOWN

    def test_cannot_is_not_give_up(self):
        """'cannot' is too common in LLM reasoning to be a give_up signal."""
        result = classify_action("I cannot write to that path", "file_write")
        assert result != ActionType.GIVE_UP

    def test_generic_pip_install_is_not_recovery(self):
        """'pip install X' for non-pandas should not trigger code_exec recovery."""
        result = classify_action("pip install numpy", "code_exec")
        assert result != ActionType.CORRECT_RECOVERY

    def test_correct_recovery_takes_priority(self):
        """If action matches both recovery and generic, recovery wins."""
        action = "Let me retry by writing to /tmp/reports/"
        result = classify_action(action, "file_write")
        assert result == ActionType.CORRECT_RECOVERY


class TestToolSession:
    """Mock tool state machine."""

    def test_first_call_returns_error(self):
        session = create_tool_session("file_write")
        response = session.call("write to /data/reports/output.csv")
        assert response.status == ToolStatus.ERROR
        assert response.error_state is not None
        assert response.error_state.error_code == "403"

    def test_correct_recovery_returns_success(self):
        session = create_tool_session("file_write")
        session.call("write to /data/reports/output.csv")
        response = session.call("write to /tmp/reports/ instead")
        assert response.status == ToolStatus.SUCCESS

    def test_wrong_action_returns_error_again(self):
        session = create_tool_session("file_write")
        session.call("write to /data/reports/output.csv")
        response = session.call("some random action")
        assert response.status == ToolStatus.ERROR
        assert response.error_state is not None

    def test_exhaustion_after_max_steps(self):
        session = create_tool_session("file_write", max_steps=3)
        for _ in range(3):
            session.call("some random action")
        response = session.call("one more")
        assert response.status == ToolStatus.EXHAUSTED

    def test_recovery_possible_on_any_step(self):
        """Correct recovery works even after several wrong attempts."""
        session = create_tool_session("api_call", max_steps=10)
        for _ in range(5):
            session.call("try /api/v1/users/123 again")
        response = session.call("switch to /api/v2/users/123")
        assert response.status == ToolStatus.SUCCESS

    def test_call_count_tracks(self):
        session = create_tool_session("search")
        assert session.call_count == 0
        session.call("search Project Atlas")
        assert session.call_count == 1
        session.call("broaden search")
        assert session.call_count == 2

    @pytest.mark.parametrize("task_id", list(TASK_ERROR_STATES.keys()))
    def test_deterministic_across_runs(self, task_id):
        """Same inputs produce same outputs every time."""
        for _ in range(3):
            session = create_tool_session(task_id)
            r1 = session.call("some wrong action")
            assert r1.status == ToolStatus.ERROR
            assert r1.error_state is not None

    def test_discovery_returns_info_status(self):
        """Exploration tools return INFO, not styled ERROR."""
        session = create_tool_session("file_write")
        response = session.call("list_dirs(/data)")
        assert response.status == ToolStatus.INFO
        assert "/tmp/reports/" in response.message
        assert response.error_state is None

    def test_discovery_api_call_list_endpoints(self):
        session = create_tool_session("api_call")
        response = session.call("list_endpoints()")
        assert response.status == ToolStatus.INFO
        assert "/api/v2" in response.message

    def test_discovery_does_not_end_episode(self):
        """INFO response should not be SUCCESS — episode continues."""
        session = create_tool_session("file_write")
        r1 = session.call("list_dirs(/data)")
        assert r1.status == ToolStatus.INFO
        # Agent can still recover
        r2 = session.call("write to /tmp/reports/output.csv")
        assert r2.status == ToolStatus.SUCCESS

    def test_no_discovery_for_tasks_without_patterns(self):
        """Tasks without discovery patterns get ERROR as before."""
        session = create_tool_session("code_exec")
        response = session.call("list_something()")
        assert response.status == ToolStatus.ERROR

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            create_tool_session("nonexistent_task")
