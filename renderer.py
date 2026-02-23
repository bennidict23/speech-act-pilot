"""Speech-act style renderer for the pilot experiment.

Renders an ErrorState into a tool output string with a specific speech-act
style. Two phases are supported:

- B1 (minimal cue): Structured JSON + [speech_act: <style>] tag + padding
- B2 (natural language): Full sentence templates per style + padding

TOKEN COUNT MATCHING IS NON-NEGOTIABLE:
All 5 styles for the same ErrorState and phase MUST produce the exact
same number of tokens (measured by tiktoken cl100k_base).

Note on tokenizer choice: cl100k_base may not match Qwen2.5's exact
tokenizer, but for within-task token parity (all styles equal for the
same task using the same tokenizer), the specific tokenizer only needs
to be internally consistent. If exact Qwen alignment is needed later,
swap TOKENIZER_NAME here.
"""
from __future__ import annotations

import json
from enum import Enum

import tiktoken

from mock_tools import ErrorState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKENIZER_NAME = "cl100k_base"

# Padding unit for B1: " ." is reliably 1 token in cl100k_base and
# semantically neutral (unlike [PAD] which is 3 tokens and breaks
# fine-grained padding). Verified empirically.
B1_PAD_UNIT = " ."

# Padding unit for B2: trailing " ." tokens (same as B1 for consistency).
B2_PAD_UNIT = " ."


class SpeechStyle(Enum):
    NEUTRAL = "neutral"
    DIAGNOSTIC = "diagnostic"
    DIRECTIVE = "directive"
    SUGGESTIVE = "suggestive"
    ACCUSATORY = "accusatory"


class Phase(Enum):
    B1 = "b1_minimal"
    B2 = "b2_natural"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return the cached tiktoken encoder."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens using the project's standard tokenizer."""
    return len(_get_encoder().encode(text))


# ---------------------------------------------------------------------------
# B1 minimal-cue renderer
# ---------------------------------------------------------------------------

def _error_state_to_json(error_state: ErrorState) -> str:
    """Serialize error state to compact JSON (excluding correct_recovery)."""
    return json.dumps(
        {
            "error_code": error_state.error_code,
            "error_type": error_state.error_type,
            "resource": error_state.resource,
            "retryable": error_state.retryable,
            "details": error_state.details,
        },
        separators=(",", ":"),
    )


def _b1_base(error_state: ErrorState, style: SpeechStyle) -> str:
    """Build the B1 base string (JSON + speech_act tag) before padding."""
    json_part = _error_state_to_json(error_state)
    tag = f" [speech_act: {style.value}]"
    return json_part + tag


def _compute_target_tokens(
    base_fn,
    error_state: ErrorState,
) -> int:
    """Compute the max token count across all 5 styles for a given base."""
    return max(
        count_tokens(base_fn(error_state, style))
        for style in SpeechStyle
    )


def _pad_to_target(text: str, target: int, pad_unit: str) -> str:
    """Pad text to exactly target tokens using the given pad unit.

    Strategy: append pad_unit repeatedly. Each pad_unit must add exactly
    1 token (verified for " ." in cl100k_base).

    Raises:
        ValueError: If exact target cannot be reached.
    """
    pad_unit_tokens = count_tokens(pad_unit)
    if pad_unit_tokens != 1:
        raise ValueError(
            f"Pad unit '{pad_unit}' is {pad_unit_tokens} tokens, "
            f"expected exactly 1."
        )

    current = count_tokens(text)
    if current > target:
        raise ValueError(
            f"Text already has {current} tokens, exceeds target {target}."
        )

    result = text
    while count_tokens(result) < target:
        result += pad_unit

    final = count_tokens(result)
    if final != target:
        raise ValueError(
            f"Padding failed: got {final} tokens, expected {target}. "
            f"Base was {current} tokens."
        )
    return result


def render_b1(error_state: ErrorState, style: SpeechStyle) -> str:
    """Render B1 minimal-cue format.

    Format: {json_blob} [speech_act: <style>] . . .
    All styles for the same error_state have EXACTLY the same token count.
    """
    base = _b1_base(error_state, style)
    target = _compute_target_tokens(_b1_base, error_state)
    return _pad_to_target(base, target, B1_PAD_UNIT)


# ---------------------------------------------------------------------------
# B2 natural-language renderer
# ---------------------------------------------------------------------------

# Templates use the same factual fields in the same order.
# CRITICAL: Templates must NOT include correct_recovery — that leaks the answer.
# "You should have anticipated" (accusatory) is a pragmatic frame, not new info.
B2_TEMPLATES: dict[SpeechStyle, str] = {
    SpeechStyle.NEUTRAL: (
        "Error {error_code}. Type: {error_type}. "
        "Resource: {resource}. "
        "{details}. "
        "Retryable: {retryable}."
    ),
    SpeechStyle.DIAGNOSTIC: (
        "Analysis indicates error {error_code}, "
        "a {error_type} condition on resource {resource}. "
        "{details}. "
        "This error is {retryable_text}."
    ),
    SpeechStyle.DIRECTIVE: (
        "ALERT: error {error_code}, {error_type} on {resource}. "
        "{details}. "
        "Do NOT repeat the failing action. "
        "This is {retryable_text}."
    ),
    SpeechStyle.SUGGESTIVE: (
        "It seems there may be an issue: error {error_code}, "
        "{error_type} involving {resource}. "
        "{details}. "
        "Perhaps this is {retryable_text}."
    ),
    SpeechStyle.ACCUSATORY: (
        "Your action caused error {error_code}, "
        "{error_type} on {resource}. "
        "{details}. "
        "You should have anticipated this. "
        "Retryable: {retryable}."
    ),
}


def _b2_base(error_state: ErrorState, style: SpeechStyle) -> str:
    """Build the B2 base string (filled template) before padding."""
    retryable_text = "retryable" if error_state.retryable else "not retryable"
    template = B2_TEMPLATES[style]
    return template.format(
        error_code=error_state.error_code,
        error_type=error_state.error_type,
        resource=error_state.resource,
        details=error_state.details,
        retryable=error_state.retryable,
        retryable_text=retryable_text,
    )


def render_b2(error_state: ErrorState, style: SpeechStyle) -> str:
    """Render B2 natural-language format.

    Full sentence templates per style, padded to identical token count.
    All styles for the same error_state have EXACTLY the same token count.
    """
    base = _b2_base(error_state, style)
    target = _compute_target_tokens(_b2_base, error_state)
    return _pad_to_target(base, target, B2_PAD_UNIT)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def render(
    error_state: ErrorState,
    style: SpeechStyle,
    phase: Phase,
) -> str:
    """Render tool output for the given error state, style, and phase.

    Args:
        error_state: The error state to render.
        style: The speech-act style.
        phase: B1 (minimal cue) or B2 (natural language).

    Returns:
        Token-count-matched string for the given style and phase.
    """
    if phase == Phase.B1:
        return render_b1(error_state, style)
    if phase == Phase.B2:
        return render_b2(error_state, style)
    raise ValueError(f"Unknown phase: {phase}")
