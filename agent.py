"""Minimal ReAct agent loop for the speech-act pilot experiment.

Drives Qwen2.5-7B-Instruct (served via vLLM) through a Thought/Action/
Observation loop against the mock tool environment. Each run produces an
immutable AgentTrajectory for post-hoc analysis.

CRITICAL CONTRACT (CC2 review):
    classify_action() must receive ONLY the Action portion of the agent's
    output. Thought text MUST be excluded — keywords in Thought (e.g.
    "retry", "impossible", "give up") cause misclassification.
    Enforcement point: parse_react_output() separates Thought from Action.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from mock_tools import ActionType, ToolStatus, classify_action, create_tool_session
from renderer import Phase, SpeechStyle, render
from tasks import TaskScenario, get_task_scenario


# ---------------------------------------------------------------------------
# Data models (immutable trajectory records)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentStep:
    """Single step in the ReAct loop."""

    step_number: int
    raw_output: str
    thought: str
    action: str  # ONLY this goes to classify_action
    observation: str
    action_type: str  # ActionType.value


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent configuration snapshot."""

    model: str
    base_url: str
    temperature: float
    max_steps: int
    seed: int


@dataclass(frozen=True)
class AgentTrajectory:
    """Complete record of a single agent run."""

    task_id: str
    style: str
    phase: str
    config: AgentConfig
    steps: tuple[AgentStep, ...]
    outcome: str  # "success" | "exhausted" | "give_up" | "max_steps"
    total_steps: int


# ---------------------------------------------------------------------------
# ReAct output parser
# ---------------------------------------------------------------------------

# Regex to find "Action:" marker (case-insensitive, handles optional whitespace)
_ACTION_RE = re.compile(r"^\s*action\s*:", re.IGNORECASE | re.MULTILINE)
_OBSERVATION_RE = re.compile(r"^\s*observation\s*:", re.IGNORECASE | re.MULTILINE)
_THOUGHT_PREFIX_RE = re.compile(r"^\s*thought\s*:\s*", re.IGNORECASE)


def parse_react_output(raw: str) -> tuple[str, str]:
    """Parse ReAct-format LLM output into (thought, action).

    CONTRACT (CC2 CRITICAL): The returned action string contains ONLY
    the Action portion. Thought text is completely excluded.
    classify_action() receives ONLY the action return value.

    Returns:
        (thought, action) tuple. action is empty string if no Action:
        marker is found.
    """
    if not raw.strip():
        return ("", "")

    # Find "Action:" marker
    action_match = _ACTION_RE.search(raw)

    if action_match is None:
        # No action marker — entire output is thought
        thought = _strip_thought_prefix(raw.strip())
        return (thought, "")

    # Split at Action: marker
    thought_raw = raw[:action_match.start()].strip()
    action_raw = raw[action_match.end():].strip()

    # Truncate action at "Observation:" if agent hallucinated one
    obs_match = _OBSERVATION_RE.search(action_raw)
    if obs_match is not None:
        action_raw = action_raw[:obs_match.start()].strip()

    thought = _strip_thought_prefix(thought_raw)
    return (thought, action_raw)


def _strip_thought_prefix(text: str) -> str:
    """Remove 'Thought:' prefix if present."""
    return _THOUGHT_PREFIX_RE.sub("", text).strip()


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(task: TaskScenario) -> str:
    """Build the ReAct system prompt for a task.

    The prompt is neutral — it does NOT mention experiments, testing,
    speech acts, or anything revealing the agent is being studied.
    """
    tool_lines = [
        f"- {t.name}({t.parameters}): {t.description}"
        for t in task.tools
    ]
    tools_block = "\n".join(tool_lines)

    return (
        "You are a helpful assistant that completes tasks using "
        "available tools.\n"
        "\n"
        f"Your goal: {task.goal}\n"
        "\n"
        f"Available tools:\n"
        f"{tools_block}\n"
        "\n"
        "Use this format for EVERY response:\n"
        "\n"
        "Thought: <your reasoning about what to do next>\n"
        "Action: <tool_name>(<arguments>)\n"
        "\n"
        "After each Action, you will receive an Observation with the "
        "result.\n"
        "Continue until the task is complete or you determine it "
        "cannot be done.\n"
        "Always respond with exactly one Thought and one Action per turn."
    )


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict[str, str]],
    config: AgentConfig,
) -> str:
    """Call the vLLM-served model via OpenAI-compatible API.

    Raises:
        RuntimeError: If the API call fails or returns empty content.
    """
    import openai

    client = openai.OpenAI(
        base_url=config.base_url,
        api_key="not-needed",
    )

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=512,
            seed=config.seed,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM API call failed: {exc}") from exc

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty content")
    return content


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

def run_agent(
    task_id: str,
    style: SpeechStyle,
    phase: Phase,
    seed: int,
    *,
    base_url: str = "http://localhost:8001/v1",
    model: str = "Qwen2.5-7B-Instruct",
    temperature: float = 0.7,
    max_steps: int = 10,
) -> AgentTrajectory:
    """Run a single ReAct agent episode.

    Returns an immutable AgentTrajectory with the complete record.
    """
    task = get_task_scenario(task_id)
    config = AgentConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_steps=max_steps,
        seed=seed,
    )
    session = create_tool_session(task_id, max_steps=max_steps)
    error_state = session.tool.error_state

    # Initial error observation (the agent's first tool call failed)
    initial_observation = render(error_state, style, phase)

    system_msg = {"role": "system", "content": build_system_prompt(task)}
    initial_user_msg = {
        "role": "user",
        "content": (
            f"I called {task.error_tool} and got this result:\n\n"
            f"{initial_observation}\n\n"
            "What should I do next?"
        ),
    }

    messages: list[dict[str, str]] = [system_msg, initial_user_msg]
    steps: list[AgentStep] = []
    outcome = "max_steps"

    for step_num in range(1, max_steps + 1):
        raw_output = call_llm(messages, config)
        thought, action = parse_react_output(raw_output)

        # CONTRACT: classify_action receives ONLY the action string.
        # NEVER pass thought, raw_output, or any combined text.
        if action:
            action_type = classify_action(action, task_id)
        else:
            action_type = ActionType.UNKNOWN

        # Get tool response (only if we have an action)
        if action:
            tool_response = session.call(action)
        else:
            tool_response = None

        # Build observation for the agent
        if tool_response is None:
            observation = (
                "No valid action was detected. "
                "Please respond with the Thought/Action format."
            )
        elif tool_response.status == ToolStatus.SUCCESS:
            observation = tool_response.message
        elif tool_response.status == ToolStatus.EXHAUSTED:
            observation = tool_response.message
        else:
            # Error — re-render with the speech-act style
            observation = render(error_state, style, phase)

        step = AgentStep(
            step_number=step_num,
            raw_output=raw_output,
            thought=thought,
            action=action,
            observation=observation,
            action_type=action_type.value,
        )
        steps.append(step)

        # Check termination
        if tool_response is not None and tool_response.status == ToolStatus.SUCCESS:
            outcome = "success"
            break
        if tool_response is not None and tool_response.status == ToolStatus.EXHAUSTED:
            outcome = "exhausted"
            break
        if action_type == ActionType.GIVE_UP:
            outcome = "give_up"
            break

        # Append to conversation for next LLM call
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}",
        })

    return AgentTrajectory(
        task_id=task_id,
        style=style.value,
        phase=phase.value,
        config=config,
        steps=tuple(steps),
        outcome=outcome,
        total_steps=len(steps),
    )
