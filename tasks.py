"""Task scenario definitions for the speech-act pilot experiment.

Each scenario specifies:
- A goal description (used in agent system prompt)
- Available tools (2-4 per task)
- Which tool triggers the error on first call
- The task_id mapping to TASK_ERROR_STATES in mock_tools.py

Tool descriptions are written as neutral API docs — no hints about
errors, recovery actions, or the experiment itself.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSchema:
    """Immutable tool description for agent system prompt."""

    name: str
    description: str
    parameters: str


@dataclass(frozen=True)
class TaskScenario:
    """Immutable task scenario definition."""

    task_id: str
    goal: str
    tools: tuple[ToolSchema, ...]
    error_tool: str


# ---------------------------------------------------------------------------
# 5 task scenarios
# ---------------------------------------------------------------------------

FILE_WRITE_TASK = TaskScenario(
    task_id="file_write",
    goal="Write the Q1 financial report to /data/reports/Q1.txt",
    tools=(
        ToolSchema(
            name="write_file",
            description="Write content to a file at the specified path.",
            parameters="path (str): file path, content (str): file content",
        ),
        ToolSchema(
            name="list_dirs",
            description="List the contents of a directory.",
            parameters="path (str): directory path",
        ),
    ),
    error_tool="write_file",
)

API_CALL_TASK = TaskScenario(
    task_id="api_call",
    goal="Fetch the profile for user 123 from /api/v1/users/123",
    tools=(
        ToolSchema(
            name="http_get",
            description="Send an HTTP GET request to the specified URL.",
            parameters="url (str): target URL",
        ),
        ToolSchema(
            name="list_endpoints",
            description="List all available API endpoints.",
            parameters="(none)",
        ),
    ),
    error_tool="http_get",
)

DB_QUERY_TASK = TaskScenario(
    task_id="db_query",
    goal="Query the table quarterly_sales for total revenue by quarter",
    tools=(
        ToolSchema(
            name="sql_query",
            description="Execute a SQL query against the database.",
            parameters="query (str): SQL query string",
        ),
        ToolSchema(
            name="list_tables",
            description="List all available tables in the database.",
            parameters="(none)",
        ),
    ),
    error_tool="sql_query",
)

SEARCH_TASK = TaskScenario(
    task_id="search",
    goal="Find the document about Project Atlas in the document store",
    tools=(
        ToolSchema(
            name="search",
            description="Search for documents matching a query string.",
            parameters="query (str): search query",
        ),
        ToolSchema(
            name="list_documents",
            description="List all documents in the document store.",
            parameters="(none)",
        ),
    ),
    error_tool="search",
)

CODE_EXEC_TASK = TaskScenario(
    task_id="code_exec",
    goal="Run the analysis script analysis.py",
    tools=(
        ToolSchema(
            name="run_code",
            description="Execute a Python script by file path.",
            parameters="script_path (str): path to .py file",
        ),
        ToolSchema(
            name="install_package",
            description="Install a Python package using pip.",
            parameters="package_name (str): name of the package to install",
        ),
    ),
    error_tool="run_code",
)

TASK_SCENARIOS: dict[str, TaskScenario] = {
    "file_write": FILE_WRITE_TASK,
    "api_call": API_CALL_TASK,
    "db_query": DB_QUERY_TASK,
    "search": SEARCH_TASK,
    "code_exec": CODE_EXEC_TASK,
}


def get_task_scenario(task_id: str) -> TaskScenario:
    """Retrieve a task scenario by ID.

    Raises:
        KeyError: If task_id is not recognized.
    """
    return TASK_SCENARIOS[task_id]


def _validate_task_ids() -> None:
    """Verify TASK_SCENARIOS keys match TASK_ERROR_STATES keys at import."""
    from mock_tools import TASK_ERROR_STATES

    scenarios = set(TASK_SCENARIOS.keys())
    errors = set(TASK_ERROR_STATES.keys())
    if scenarios != errors:
        raise RuntimeError(
            f"Task ID mismatch: scenarios={scenarios}, errors={errors}"
        )


_validate_task_ids()
