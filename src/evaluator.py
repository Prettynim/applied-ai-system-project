"""
Reliability evaluation suite for the Music Recommender.

Runs a structured test suite that measures:
- Recommendation correctness  (does the right genre appear at #1 for clear profiles?)
- Guardrail effectiveness     (do known edge cases trigger the expected warnings?)
- System determinism          (same input always produces same output?)
- Confidence score range      (are scores reasonable across all profile types?)

Each test case has a profile, a description, and one or more pass/fail checks.
Results are collected into an EvaluationReport with an overall reliability score.

Usage:
    python -m src.evaluator
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple

try:
    from recommender import load_songs, recommend_songs
    from guardrails import run_guardrails, Severity, GuardrailIssue
except ImportError:
    from src.recommender import load_songs, recommend_songs
    from src.guardrails import run_guardrails, Severity, GuardrailIssue

_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CheckFn = Callable[[List, List[GuardrailIssue]], Tuple[bool, str]]


@dataclass
class TestCase:
    name: str
    profile: Dict
    description: str
    checks: List[CheckFn]


@dataclass
class CheckOutcome:
    check_name: str
    passed: bool
    message: str


@dataclass
class CaseResult:
    name: str
    description: str
    passed: bool
    outcomes: List[CheckOutcome]
    top_title: str
    top_score: float
    confidence: float
    guardrail_count: int


@dataclass
class EvaluationReport:
    total_tests: int
    passed_tests: int
    failed_tests: int
    reliability_score: float
    results: List[CaseResult]


# ---------------------------------------------------------------------------
# Confidence scoring (rule-based, no API needed)
# ---------------------------------------------------------------------------

def confidence_score(score: float, max_score: float = 6.5) -> float:
    """Normalizes a raw recommendation score to a 0.0-1.0 confidence value."""
    return round(min(max(score / max_score, 0.0), 1.0), 3)


# ---------------------------------------------------------------------------
# Check factories
# ---------------------------------------------------------------------------

def check_top1_genre(expected_genre: str) -> CheckFn:
    """Top recommendation must match the given genre."""
    def _check(results, issues):
        if not results:
            return False, "No results returned"
        actual = results[0][0]["genre"]
        passed = actual == expected_genre
        return passed, f"top-1 genre={actual!r} (expected {expected_genre!r})"
    _check.__name__ = f"top1_genre_is_{expected_genre}"
    return _check


def check_guardrail_fires(code: str) -> CheckFn:
    """A guardrail with the given code must appear in the issue list."""
    def _check(results, issues):
        found = [i.code for i in issues]
        passed = code in found
        return passed, f"expected guardrail {code!r}, found: {found}"
    _check.__name__ = f"guardrail_{code}_fires"
    return _check


def check_no_errors(results, issues) -> Tuple[bool, str]:
    """No ERROR-severity guardrails must be raised."""
    errors = [i.code for i in issues if i.severity == Severity.ERROR]
    return len(errors) == 0, f"error guardrails: {errors}"


check_no_errors.__name__ = "no_error_guardrails"


def check_result_count(k: int) -> CheckFn:
    """Exactly k results must be returned."""
    def _check(results, issues):
        passed = len(results) == k
        return passed, f"expected {k} results, got {len(results)}"
    _check.__name__ = f"returns_{k}_results"
    return _check


def check_top_score_above(threshold: float) -> CheckFn:
    """Top result score must exceed the given threshold."""
    def _check(results, issues):
        if not results:
            return False, "No results"
        score = results[0][1]
        passed = score >= threshold
        return passed, f"top score={score:.2f} (required >= {threshold})"
    _check.__name__ = f"top_score_above_{threshold}"
    return _check


def check_top_score_below(threshold: float) -> CheckFn:
    """Top result score must be below threshold (detects inflated scores on bad profiles)."""
    def _check(results, issues):
        if not results:
            return True, "No results"
        score = results[0][1]
        passed = score < threshold
        return passed, f"top score={score:.2f} (required < {threshold})"
    _check.__name__ = f"top_score_below_{threshold}"
    return _check


# ---------------------------------------------------------------------------
# Test suite definition
# ---------------------------------------------------------------------------

def build_test_suite() -> List[TestCase]:
    """
    Returns the standard reliability test cases.

    Tests T01-T03: Normal profiles — clear genre/mood/energy matches.
    Tests T04-T05: Catalog gap profiles — genre or mood missing from catalog.
    Tests T06-T07: Adversarial profiles — contradictions or edge-case energy.
    Test  T08    : Valence anchor bias check.
    """
    return [
        TestCase(
            name="T01 - Pop / Happy (Normal)",
            profile={"genre": "pop", "mood": "happy", "target_energy": 0.8, "likes_acoustic": False},
            description="Clear pop profile. Catalog has 2 pop songs; #1 should be pop.",
            checks=[
                check_top1_genre("pop"),
                check_no_errors,
                check_result_count(5),
                check_top_score_above(3.5),
            ],
        ),
        TestCase(
            name="T02 - Lofi Study (Normal)",
            profile={"genre": "lofi", "mood": "focused", "target_energy": 0.35, "likes_acoustic": True},
            description="Classic study profile. Catalog has 3 lofi songs; #1 should be lofi.",
            checks=[
                check_top1_genre("lofi"),
                check_no_errors,
                check_result_count(5),
                check_top_score_above(4.0),
            ],
        ),
        TestCase(
            name="T03 - Rock Workout (Normal)",
            profile={"genre": "rock", "mood": "intense", "target_energy": 0.92, "likes_acoustic": False},
            description="High-energy workout profile. Catalog has 1 rock song; it should rank #1.",
            checks=[
                check_top1_genre("rock"),
                check_no_errors,
                check_result_count(5),
                check_top_score_above(3.0),
            ],
        ),
        TestCase(
            name="T04 - Unknown Genre (Catalog Gap)",
            profile={"genre": "reggae", "mood": "happy", "target_energy": 0.65, "likes_acoustic": False},
            description=(
                "Reggae is not in the catalog. The GENRE_NOT_IN_CATALOG guardrail must fire. "
                "No results should still be returned (graceful degradation)."
            ),
            checks=[
                check_guardrail_fires("GENRE_NOT_IN_CATALOG"),
                check_no_errors,
                check_result_count(5),
            ],
        ),
        TestCase(
            name="T05 - Impossible Mood (Catalog Gap)",
            profile={"genre": "lofi", "mood": "wistful", "target_energy": 0.40, "likes_acoustic": True},
            description=(
                "'wistful' is not in the catalog. MOOD_NOT_IN_CATALOG guardrail must fire. "
                "Genre match should still surface lofi songs in results."
            ),
            checks=[
                check_guardrail_fires("MOOD_NOT_IN_CATALOG"),
                check_top1_genre("lofi"),
                check_result_count(5),
            ],
        ),
        TestCase(
            name="T06 - High-Energy Acoustic (Adversarial)",
            profile={"genre": "folk", "mood": "intense", "target_energy": 0.90, "likes_acoustic": True},
            description=(
                "Contradictory: high energy + acoustic preference. "
                "HIGH_ENERGY_ACOUSTIC_CONFLICT must fire. "
                "Documented failure mode: quiet folk song may rank #1 despite energy mismatch."
            ),
            checks=[
                check_guardrail_fires("HIGH_ENERGY_ACOUSTIC_CONFLICT"),
                check_no_errors,
                check_result_count(5),
            ],
        ),
        TestCase(
            name="T07 - Dead-Center Energy (Adversarial)",
            profile={"genre": "synthwave", "mood": "moody", "target_energy": 0.50, "likes_acoustic": False},
            description=(
                "Energy=0.5 sits in a catalog gap (most songs cluster at <0.4 or >0.7). "
                "Top score should still be reasonable even without strong energy match."
            ),
            checks=[
                check_no_errors,
                check_result_count(5),
                check_top_score_above(2.0),
                check_top_score_below(6.0),
            ],
        ),
        TestCase(
            name="T08 - EDM Dance Floor (Normal)",
            profile={"genre": "edm", "mood": "euphoric", "target_energy": 0.95, "likes_acoustic": False},
            description=(
                "High-energy EDM profile. Catalog has 1 EDM song; it should dominate. "
                "Also checks that top score is high but not inflated past the max."
            ),
            checks=[
                check_top1_genre("edm"),
                check_no_errors,
                check_result_count(5),
                check_top_score_above(3.0),
                check_top_score_below(6.6),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test_suite(songs: List[Dict], k: int = 5) -> EvaluationReport:
    """
    Runs all test cases and returns an EvaluationReport.

    Also appends a determinism test (T09) that verifies the rule engine
    produces identical rankings when called twice with the same input.
    """
    test_cases = build_test_suite()
    results: List[CaseResult] = []
    passed_count = 0

    for tc in test_cases:
        issues = run_guardrails(tc.profile, songs)
        recommendations = recommend_songs(tc.profile, songs, k=k)

        outcomes: List[CheckOutcome] = []
        test_passed = True
        for fn in tc.checks:
            ok, msg = fn(recommendations, issues)
            outcomes.append(CheckOutcome(check_name=fn.__name__, passed=ok, message=msg))
            if not ok:
                test_passed = False

        top_score = recommendations[0][1] if recommendations else 0.0
        top_title = recommendations[0][0]["title"] if recommendations else "N/A"

        results.append(CaseResult(
            name=tc.name,
            description=tc.description,
            passed=test_passed,
            outcomes=outcomes,
            top_title=top_title,
            top_score=top_score,
            confidence=confidence_score(top_score),
            guardrail_count=len(issues),
        ))
        if test_passed:
            passed_count += 1

    # --- T09: Determinism ---
    ref_profile = {"genre": "jazz", "mood": "relaxed", "target_energy": 0.35, "likes_acoustic": True}
    run1 = [r[0]["title"] for r in recommend_songs(ref_profile, songs, k=k)]
    run2 = [r[0]["title"] for r in recommend_songs(ref_profile, songs, k=k)]
    det_passed = run1 == run2
    det_msg = "identical" if det_passed else f"run1={run1} run2={run2}"

    results.append(CaseResult(
        name="T09 - Determinism (Rule Engine)",
        description=(
            "Same profile run twice must produce identical rankings. "
            "The rule-based engine is deterministic by design."
        ),
        passed=det_passed,
        outcomes=[CheckOutcome("same_output_on_repeat_call", det_passed, det_msg)],
        top_title=run1[0] if run1 else "N/A",
        top_score=0.0,
        confidence=0.0,
        guardrail_count=0,
    ))
    if det_passed:
        passed_count += 1

    total = len(results)
    return EvaluationReport(
        total_tests=total,
        passed_tests=passed_count,
        failed_tests=total - passed_count,
        reliability_score=round(passed_count / total, 3),
        results=results,
    )


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(report: EvaluationReport) -> None:
    """Prints the evaluation report in a readable tabular format."""
    width = 70
    sep = "=" * width
    thin = "-" * width

    print(f"\n{sep}")
    print("  RELIABILITY EVALUATION REPORT")
    print(sep)

    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n  [{status}] {result.name}")
        print(f"         {result.description}")
        if result.top_score > 0:
            print(
                f"         Top result : {result.top_title!r} "
                f"(score={result.top_score:.2f}/6.5, confidence={result.confidence:.0%})"
            )
        if result.guardrail_count > 0:
            print(f"         Guardrails : {result.guardrail_count} issue(s) raised")
        for outcome in result.outcomes:
            tick = "ok" if outcome.passed else "!!"
            print(f"         [{tick}] {outcome.check_name}: {outcome.message}")

    print(f"\n{thin}")
    print(f"  Results           : {report.passed_tests}/{report.total_tests} passed")
    print(f"  Reliability score : {report.reliability_score:.0%}")

    if report.reliability_score >= 0.9:
        grade = "EXCELLENT"
    elif report.reliability_score >= 0.75:
        grade = "GOOD"
    elif report.reliability_score >= 0.6:
        grade = "FAIR"
    else:
        grade = "NEEDS IMPROVEMENT"

    print(f"  Grade             : {grade}")
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    songs = load_songs(_SONGS_PATH)
    print(f"Catalog loaded: {len(songs)} songs")
    report = run_test_suite(songs)
    print_report(report)


if __name__ == "__main__":
    main()
