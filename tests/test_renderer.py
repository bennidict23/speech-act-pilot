"""Tests for renderer.py — token parity, no info leakage, style correctness.

Token count matching is NON-NEGOTIABLE. If any parity test fails,
the experiment is invalid.
"""
from __future__ import annotations

import pytest

from mock_tools import TASK_ERROR_STATES
from renderer import (
    Phase,
    SpeechStyle,
    count_tokens,
    render,
    render_b1,
    render_b2,
)

TASK_IDS = list(TASK_ERROR_STATES.keys())


# ---------------------------------------------------------------------------
# Token parity (THE critical tests)
# ---------------------------------------------------------------------------


class TestB1TokenParity:
    """All 5 styles for the same task MUST have identical B1 token counts."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_token_parity(self, task_id):
        error_state = TASK_ERROR_STATES[task_id]
        counts = {
            style.value: count_tokens(render_b1(error_state, style))
            for style in SpeechStyle
        }
        unique = set(counts.values())
        assert len(unique) == 1, f"B1 token mismatch for {task_id}: {counts}"

    def test_all_25_combinations(self):
        """Exhaustive check: 5 tasks x 5 styles = 25 B1 renderings."""
        for task_id, error_state in TASK_ERROR_STATES.items():
            token_counts = [
                count_tokens(render_b1(error_state, s)) for s in SpeechStyle
            ]
            assert len(set(token_counts)) == 1, (
                f"B1 parity failed for {task_id}: {token_counts}"
            )


class TestB2TokenParity:
    """All 5 styles for the same task MUST have identical B2 token counts."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_token_parity(self, task_id):
        error_state = TASK_ERROR_STATES[task_id]
        counts = {
            style.value: count_tokens(render_b2(error_state, style))
            for style in SpeechStyle
        }
        unique = set(counts.values())
        assert len(unique) == 1, f"B2 token mismatch for {task_id}: {counts}"

    def test_all_25_combinations(self):
        """Exhaustive check: 5 tasks x 5 styles = 25 B2 renderings."""
        for task_id, error_state in TASK_ERROR_STATES.items():
            token_counts = [
                count_tokens(render_b2(error_state, s)) for s in SpeechStyle
            ]
            assert len(set(token_counts)) == 1, (
                f"B2 parity failed for {task_id}: {token_counts}"
            )


# ---------------------------------------------------------------------------
# No information leakage
# ---------------------------------------------------------------------------


class TestNoInfoLeakage:
    """Style templates must NOT introduce facts beyond error state S."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_b1_no_recovery_hint(self, task_id):
        """B1 output must not contain the correct_recovery value."""
        error_state = TASK_ERROR_STATES[task_id]
        for style in SpeechStyle:
            rendered = render_b1(error_state, style)
            assert error_state.correct_recovery not in rendered, (
                f"B1 leaks correct_recovery for {task_id}/{style.value}"
            )

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_b2_no_recovery_hint(self, task_id):
        """B2 output must not contain the correct_recovery value."""
        error_state = TASK_ERROR_STATES[task_id]
        for style in SpeechStyle:
            rendered = render_b2(error_state, style)
            assert error_state.correct_recovery not in rendered, (
                f"B2 leaks correct_recovery for {task_id}/{style.value}"
            )

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_b2_contains_required_facts(self, task_id):
        """All B2 styles must contain error_code, resource, and details."""
        error_state = TASK_ERROR_STATES[task_id]
        for style in SpeechStyle:
            rendered = render_b2(error_state, style)
            assert error_state.error_code in rendered, (
                f"B2 missing error_code for {task_id}/{style.value}"
            )
            assert error_state.resource in rendered, (
                f"B2 missing resource for {task_id}/{style.value}"
            )
            assert error_state.details in rendered, (
                f"B2 missing details for {task_id}/{style.value}"
            )


# ---------------------------------------------------------------------------
# Style correctness
# ---------------------------------------------------------------------------


class TestStyleCorrectness:
    """Verify each style has its distinguishing characteristics."""

    @pytest.mark.parametrize("task_id", TASK_IDS)
    def test_b1_contains_style_tag(self, task_id):
        error_state = TASK_ERROR_STATES[task_id]
        for style in SpeechStyle:
            rendered = render_b1(error_state, style)
            assert f"[speech_act: {style.value}]" in rendered

    def test_b1_contains_json_fields(self):
        error_state = TASK_ERROR_STATES["file_write"]
        rendered = render_b1(error_state, SpeechStyle.NEUTRAL)
        assert '"error_code"' in rendered
        assert '"403"' in rendered

    def test_b2_styles_produce_distinct_text(self):
        """B2 styles should have different text (not just padding)."""
        error_state = TASK_ERROR_STATES["file_write"]
        stripped = {
            style: render_b2(error_state, style).rstrip(" .")
            for style in SpeechStyle
        }
        unique = set(stripped.values())
        assert len(unique) == 5, "B2 styles should produce 5 distinct texts"

    def test_b1_json_identical_across_styles(self):
        """B1 JSON portion (before the tag) must be identical across styles."""
        error_state = TASK_ERROR_STATES["api_call"]
        json_parts = set()
        for style in SpeechStyle:
            rendered = render_b1(error_state, style)
            json_part = rendered.split(" [speech_act:")[0]
            json_parts.add(json_part)
        assert len(json_parts) == 1, "B1 JSON differs across styles"


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


class TestRenderDispatch:
    """The render() function correctly dispatches to B1/B2."""

    def test_b1_dispatch(self):
        error_state = TASK_ERROR_STATES["file_write"]
        result = render(error_state, SpeechStyle.NEUTRAL, Phase.B1)
        assert "[speech_act:" in result

    def test_b2_dispatch(self):
        error_state = TASK_ERROR_STATES["file_write"]
        result = render(error_state, SpeechStyle.NEUTRAL, Phase.B2)
        assert "[speech_act:" not in result

    def test_invalid_phase_raises(self):
        error_state = TASK_ERROR_STATES["file_write"]
        with pytest.raises(ValueError, match="Unknown phase"):
            render(error_state, SpeechStyle.NEUTRAL, "invalid")  # type: ignore[arg-type]
