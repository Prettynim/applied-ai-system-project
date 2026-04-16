"""
Tests for src/evaluator.py

Covers:
- confidence_score() normalizes correctly
- run_test_suite() returns correct counts and structure
- EvaluationReport reliability score is in [0, 1]
- Determinism test (T09) always passes for the rule engine
- Known-good profiles pass their checks
- Known-edge-case profiles pass their guardrail checks
"""

import pytest
from src.evaluator import (
    confidence_score,
    run_test_suite,
    build_test_suite,
    EvaluationReport,
    CaseResult,
)
from src.recommender import load_songs
import os

_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


@pytest.fixture(scope="module")
def songs():
    return load_songs(_SONGS_PATH)


@pytest.fixture(scope="module")
def report(songs):
    return run_test_suite(songs)


# ---------------------------------------------------------------------------
# confidence_score()
# ---------------------------------------------------------------------------

def test_confidence_score_full():
    assert confidence_score(6.5) == 1.0


def test_confidence_score_zero():
    assert confidence_score(0.0) == 0.0


def test_confidence_score_half():
    result = confidence_score(3.25)
    assert abs(result - 0.5) < 0.01


def test_confidence_score_over_max_clamps_to_1():
    assert confidence_score(10.0) == 1.0


def test_confidence_score_negative_clamps_to_0():
    assert confidence_score(-1.0) == 0.0


# ---------------------------------------------------------------------------
# EvaluationReport structure
# ---------------------------------------------------------------------------

def test_report_has_results(report):
    assert len(report.results) > 0


def test_report_total_equals_len_results(report):
    assert report.total_tests == len(report.results)


def test_report_passed_plus_failed_equals_total(report):
    assert report.passed_tests + report.failed_tests == report.total_tests


def test_reliability_score_in_range(report):
    assert 0.0 <= report.reliability_score <= 1.0


def test_reliability_score_consistent_with_counts(report):
    expected = round(report.passed_tests / report.total_tests, 3)
    assert report.reliability_score == expected


# ---------------------------------------------------------------------------
# Individual test case results
# ---------------------------------------------------------------------------

def _find_result(report: EvaluationReport, name_prefix: str) -> CaseResult:
    for r in report.results:
        if r.name.startswith(name_prefix):
            return r
    raise KeyError(f"No test result starting with {name_prefix!r}")


def test_t01_pop_happy_passes(report):
    result = _find_result(report, "T01")
    assert result.passed, f"T01 failed: {[o for o in result.outcomes if not o.passed]}"


def test_t02_lofi_study_passes(report):
    result = _find_result(report, "T02")
    assert result.passed, f"T02 failed: {[o for o in result.outcomes if not o.passed]}"


def test_t03_rock_workout_passes(report):
    result = _find_result(report, "T03")
    assert result.passed, f"T03 failed: {[o for o in result.outcomes if not o.passed]}"


def test_t04_unknown_genre_passes(report):
    # Edge case: guardrail check, not genre-in-top1
    result = _find_result(report, "T04")
    assert result.passed, f"T04 failed: {[o for o in result.outcomes if not o.passed]}"


def test_t05_impossible_mood_passes(report):
    result = _find_result(report, "T05")
    assert result.passed, f"T05 failed: {[o for o in result.outcomes if not o.passed]}"


def test_t09_determinism_always_passes(report):
    result = _find_result(report, "T09")
    assert result.passed, "Rule engine is non-deterministic — this should never happen"


# ---------------------------------------------------------------------------
# Test suite completeness
# ---------------------------------------------------------------------------

def test_suite_has_at_least_8_manual_cases():
    suite = build_test_suite()
    assert len(suite) >= 8


def test_all_test_cases_have_checks():
    suite = build_test_suite()
    for tc in suite:
        assert len(tc.checks) >= 1, f"{tc.name} has no checks"


def test_all_results_have_outcomes(report):
    for result in report.results:
        assert len(result.outcomes) >= 1, f"{result.name} has no outcomes"


def test_all_outcomes_have_messages(report):
    for result in report.results:
        for outcome in result.outcomes:
            assert isinstance(outcome.message, str)
            assert len(outcome.message) > 0
