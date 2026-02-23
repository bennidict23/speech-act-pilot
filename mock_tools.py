"""Deterministic mock tool environment for the speech-act pilot experiment.

Each mock tool follows a fixed error→recovery state machine:
- First call: returns structured error state S
- Correct recovery action: returns success
- Wrong action: returns error state again
- Exceeds max steps: returns exhausted

Action classification is RULE-BASED (string matching), never LLM-based.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Data models (all immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorState:
    """Immutable error state returned by a mock tool on first call."""

    error_code: str
    error_type: str
    resource: str
    retryable: bool
    correct_recovery: str
    details: str


class ToolStatus(Enum):
    ERROR = "error"
    SUCCESS = "success"
    EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class ToolResponse:
    """Immutable response from a mock tool call."""

    status: ToolStatus
    message: str
    error_state: ErrorState | None = None


# ---------------------------------------------------------------------------
# 5 task error states
# ---------------------------------------------------------------------------

FILE_WRITE_ERROR = ErrorState(
    error_code="403",
    error_type="permission_denied",
    resource="/data/reports/",
    retryable=False,
    correct_recovery="use_alternate_path",
    details="Write permission denied for /data/reports/",
)

API_CALL_ERROR = ErrorState(
    error_code="404",
    error_type="not_found",
    resource="/api/v1/users/123",
    retryable=False,
    correct_recovery="use_v2_endpoint",
    details="Endpoint /api/v1/users/123 not found",
)

DB_QUERY_ERROR = ErrorState(
    error_code="TABLE_NOT_FOUND",
    error_type="table_not_found",
    resource="quarterly_sales",
    retryable=False,
    correct_recovery="list_tables_first",
    details="Table 'quarterly_sales' does not exist in database",
)

SEARCH_ERROR = ErrorState(
    error_code="NO_RESULTS",
    error_type="no_results",
    resource="Project Atlas",
    retryable=False,
    correct_recovery="broaden_query",
    details="Search for 'Project Atlas' returned 0 results",
)

CODE_EXEC_ERROR = ErrorState(
    error_code="MODULE_NOT_FOUND",
    error_type="import_error",
    resource="pandas",
    retryable=False,
    correct_recovery="install_module",
    details="ModuleNotFoundError: No module named 'pandas'",
)

TASK_ERROR_STATES: dict[str, ErrorState] = {
    "file_write": FILE_WRITE_ERROR,
    "api_call": API_CALL_ERROR,
    "db_query": DB_QUERY_ERROR,
    "search": SEARCH_ERROR,
    "code_exec": CODE_EXEC_ERROR,
}


# ---------------------------------------------------------------------------
# Action classification (rule-based only)
# ---------------------------------------------------------------------------

class ActionType(Enum):
    RETRY = "retry"
    MODIFY_PARAMS = "modify_params"
    SWITCH_TOOL = "switch_tool"
    ASK_USER = "ask_user"
    GIVE_UP = "give_up"
    CORRECT_RECOVERY = "correct_recovery"
    UNKNOWN = "unknown"


# Per-task patterns that indicate the correct recovery action.
# These are checked via substring match on lowercased action text.
_RECOVERY_PATTERNS: dict[str, list[str]] = {
    "file_write": [
        "/tmp/reports", "tmp/reports", "alternate path",
        "different directory", "different path", "another directory",
        "writable path", "writable directory",
    ],
    "api_call": [
        "/api/v2", "v2/users", "version 2", "v2 endpoint",
        "api/v2", "endpoint v2",
    ],
    "db_query": [
        "list_tables", "show tables", "list tables",
        "available tables", "show_tables", "describe tables",
    ],
    "search": [
        "broaden", "shorter query", "just atlas",
        '"atlas"', "search atlas", "'atlas'",
        "widen", "broader",
    ],
    "code_exec": [
        "install pandas", "pip install", "install module",
        "install_module", "install package",
    ],
}

_GIVE_UP_KEYWORDS = ["give up", "cannot", "impossible", "abort", "give_up"]
_ASK_USER_KEYWORDS = ["ask user", "ask_user", "need help", "request assistance"]
_SWITCH_TOOL_KEYWORDS = ["switch tool", "different tool", "use_tool", "alternative tool"]
_RETRY_KEYWORDS = ["retry", "try again", "same action", "repeat"]
_MODIFY_KEYWORDS = ["modify", "change param", "update param", "adjust param"]


def classify_action(action_text: str, task_id: str) -> ActionType:
    """Rule-based action classification. NO LLM calls.

    Checks task-specific correct recovery patterns first (most specific),
    then falls back to generic action categories.
    """
    lower = action_text.lower()

    # 1. Task-specific correct recovery (most specific)
    for pattern in _RECOVERY_PATTERNS.get(task_id, []):
        if pattern in lower:
            return ActionType.CORRECT_RECOVERY

    # 2. Generic categories (ordered: specific → general)
    if any(kw in lower for kw in _GIVE_UP_KEYWORDS):
        return ActionType.GIVE_UP
    if any(kw in lower for kw in _ASK_USER_KEYWORDS):
        return ActionType.ASK_USER
    if any(kw in lower for kw in _SWITCH_TOOL_KEYWORDS):
        return ActionType.SWITCH_TOOL
    if any(kw in lower for kw in _RETRY_KEYWORDS):
        return ActionType.RETRY
    if any(kw in lower for kw in _MODIFY_KEYWORDS):
        return ActionType.MODIFY_PARAMS

    return ActionType.UNKNOWN


# ---------------------------------------------------------------------------
# Mock tool session (state machine)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MockTool:
    """Immutable mock tool definition."""

    task_id: str
    error_state: ErrorState
    max_steps: int = 10


def _success_message(task_id: str) -> str:
    """Return the success message for a given task."""
    messages = {
        "file_write": (
            "File successfully written to /tmp/reports/output.csv"
        ),
        "api_call": (
            "API call to /api/v2/users/123 returned 200 OK with user data"
        ),
        "db_query": (
            "list_tables returned: "
            "['monthly_sales', 'annual_sales', 'quarterly_revenue']"
        ),
        "search": "Search for 'Atlas' returned 3 results",
        "code_exec": (
            "pandas installed successfully. Module ready for import."
        ),
    }
    return messages.get(task_id, "Action completed successfully.")


class ToolSession:
    """Mutable session state for a single tool interaction.

    Tracks call count as session state. The underlying MockTool and
    ErrorState remain immutable.
    """

    def __init__(self, tool: MockTool) -> None:
        self._tool = tool
        self._call_count = 0

    @property
    def tool(self) -> MockTool:
        return self._tool

    @property
    def call_count(self) -> int:
        return self._call_count

    def call(self, action_text: str) -> ToolResponse:
        """Process an agent action. Returns immutable ToolResponse."""
        self._call_count += 1

        if self._call_count > self._tool.max_steps:
            return ToolResponse(
                status=ToolStatus.EXHAUSTED,
                message=(
                    f"Maximum steps ({self._tool.max_steps}) exceeded."
                ),
            )

        action_type = classify_action(action_text, self._tool.task_id)

        if action_type == ActionType.CORRECT_RECOVERY:
            return ToolResponse(
                status=ToolStatus.SUCCESS,
                message=_success_message(self._tool.task_id),
            )

        return ToolResponse(
            status=ToolStatus.ERROR,
            message="Action did not resolve the error.",
            error_state=self._tool.error_state,
        )


def create_tool_session(
    task_id: str,
    max_steps: int = 10,
) -> ToolSession:
    """Create a new tool session for a task.

    Args:
        task_id: One of the keys in TASK_ERROR_STATES.
        max_steps: Maximum number of agent actions before exhaustion.

    Returns:
        A fresh ToolSession ready for agent interaction.

    Raises:
        KeyError: If task_id is not recognized.
    """
    error_state = TASK_ERROR_STATES[task_id]
    tool = MockTool(task_id=task_id, error_state=error_state, max_steps=max_steps)
    return ToolSession(tool)
