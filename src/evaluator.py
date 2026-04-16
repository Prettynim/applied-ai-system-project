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

import os      # os.path for resolving the songs.csv path
import sys     # sys.path manipulation for the dual-import pattern
from dataclasses import dataclass           # lightweight data containers for test results
from typing import List, Dict, Callable, Tuple   # type hints for test functions and results

# Dual import path: works both as `python -m src.evaluator` and `python evaluator.py`
try:
    from recommender import load_songs, recommend_songs        # rule-based scoring engine
    from guardrails import run_guardrails, Severity, GuardrailIssue  # pre-flight checks
except ImportError:
    from src.recommender import load_songs, recommend_songs
    from src.guardrails import run_guardrails, Severity, GuardrailIssue

# Resolve songs.csv relative to this file — same pattern as agent.py and rag.py
_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

# CheckFn is the type of a single pass/fail check function.
# It receives the recommendation results and guardrail issues, returns (passed, message).
CheckFn = Callable[[List, List[GuardrailIssue]], Tuple[bool, str]]


@dataclass
class TestCase:
    """One predefined test case: a profile, a description, and a list of checks to run."""
    name: str              # short identifier shown in the report (e.g., "T01 - Pop / Happy")
    profile: Dict          # user preference dict passed to the scoring engine
    description: str       # explains what this test is verifying and why it matters
    checks: List[CheckFn]  # list of check functions — ALL must pass for the case to pass


@dataclass
class CheckOutcome:
    """Result of running a single check function within a test case."""
    check_name: str   # function's __name__ attribute — used in the report
    passed: bool      # True if the check returned True
    message: str      # human-readable detail from the check function


@dataclass
class CaseResult:
    """Aggregated result for one complete test case (all checks run)."""
    name: str
    description: str
    passed: bool              # True only if ALL checks passed
    outcomes: List[CheckOutcome]
    top_title: str            # title of the #1 recommended song
    top_score: float          # raw score of the #1 result (out of 6.5)
    confidence: float         # top_score normalized to 0.0–1.0
    guardrail_count: int      # number of guardrail issues raised for this profile


@dataclass
class EvaluationReport:
    """Summary of the entire test run — printed at the end of the report."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    reliability_score: float   # passed_tests / total_tests, rounded to 3 decimal places
    results: List[CaseResult]


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def confidence_score(score: float, max_score: float = 6.5) -> float:
    """
    Normalizes a raw recommendation score to a 0.0–1.0 confidence value.
    max() and min() clamp the result to [0, 1] in case of floating-point edge cases.
    Used to give a percentage "how well did the system serve this profile?" metric.
    """
    return round(min(max(score / max_score, 0.0), 1.0), 3)


# ---------------------------------------------------------------------------
# Check factory functions
# ---------------------------------------------------------------------------

def check_top1_genre(expected_genre: str) -> CheckFn:
    """
    Returns a check function that passes only if the #1 recommendation matches expected_genre.
    The genre field of the top song is compared as an exact string.
    """
    def _check(results, issues):
        if not results:
            return False, "No results returned"
        actual = results[0][0]["genre"]           # results[0] = top song tuple; [0] = song dict; ["genre"] = genre
        passed = actual == expected_genre
        return passed, f"top-1 genre={actual!r} (expected {expected_genre!r})"
    _check.__name__ = f"top1_genre_is_{expected_genre}"   # used in the report output
    return _check


def check_guardrail_fires(code: str) -> CheckFn:
    """
    Returns a check function that passes only if a guardrail with the given code
    appears in the issues list. Used to verify that edge cases trigger the expected warnings.
    """
    def _check(results, issues):
        found = [i.code for i in issues]   # extract all issue codes as a list
        passed = code in found
        return passed, f"expected guardrail {code!r}, found: {found}"
    _check.__name__ = f"guardrail_{code}_fires"
    return _check


def check_no_errors(results, issues) -> Tuple[bool, str]:
    """
    Standalone check (not a factory) — passes only if no ERROR-severity guardrails were raised.
    WARNING and INFO issues are acceptable; ERROR means invalid input reached the engine.
    """
    errors = [i.code for i in issues if i.severity == Severity.ERROR]
    return len(errors) == 0, f"error guardrails: {errors}"

# Assign a readable __name__ so it shows correctly in the report
check_no_errors.__name__ = "no_error_guardrails"


def check_result_count(k: int) -> CheckFn:
    """Returns a check function that passes only if exactly k results were returned."""
    def _check(results, issues):
        passed = len(results) == k
        return passed, f"expected {k} results, got {len(results)}"
    _check.__name__ = f"returns_{k}_results"
    return _check


def check_top_score_above(threshold: float) -> CheckFn:
    """
    Returns a check function that passes only if the top result's score >= threshold.
    Used to verify that clear profiles produce strong matches (guards against score regression).
    """
    def _check(results, issues):
        if not results:
            return False, "No results"
        score = results[0][1]    # results[0] = top song tuple; [1] = score float
        passed = score >= threshold
        return passed, f"top score={score:.2f} (required >= {threshold})"
    _check.__name__ = f"top_score_above_{threshold}"
    return _check


def check_top_score_below(threshold: float) -> CheckFn:
    """
    Returns a check function that passes only if the top result's score < threshold.
    Used to detect inflated scores on adversarial profiles and enforce the 6.5-point max.
    """
    def _check(results, issues):
        if not results:
            return True, "No results"   # no results can't have an inflated score
        score = results[0][1]
        passed = score < threshold
        return passed, f"top score={score:.2f} (required < {threshold})"
    _check.__name__ = f"top_score_below_{threshold}"
    return _check


# ---------------------------------------------------------------------------
# Test suite definition (T01–T08)
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
                check_top1_genre("pop"),      # top result must be a pop song
                check_no_errors,              # no invalid input errors
                check_result_count(5),        # must return exactly 5 results
                check_top_score_above(3.5),   # pop+happy+energy match should score well
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
                check_top_score_above(4.0),   # lofi+focused+acoustic earns 4 out of the possible 6.5
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
                check_top_score_above(3.0),   # lower threshold because catalog has only 1 rock song
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
                check_guardrail_fires("GENRE_NOT_IN_CATALOG"),  # guardrail must detect the gap
                check_no_errors,                                  # gap is a WARNING, not an ERROR
                check_result_count(5),                            # system must still return 5 results
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
                check_guardrail_fires("MOOD_NOT_IN_CATALOG"),  # mood gap must be detected
                check_top1_genre("lofi"),    # genre bonus still works even without mood bonus
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
                check_guardrail_fires("HIGH_ENERGY_ACOUSTIC_CONFLICT"),  # contradiction must be detected
                check_no_errors,    # contradiction is a WARNING, not an ERROR
                check_result_count(5),
                # NOTE: no top1_genre check here — the energy mismatch means #1 may not be folk
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
                check_top_score_above(2.0),   # genre+mood match should still produce a reasonable score
                check_top_score_below(6.0),   # energy gap means top score can't be near-perfect
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
                check_top_score_below(6.6),   # 6.6 > 6.5 max — catches any floating-point overflow
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test_suite(songs: List[Dict], k: int = 5) -> EvaluationReport:
    """
    Runs all test cases and returns an EvaluationReport.

    Also appends a determinism test (T09) that verifies the rule engine
    produces identical rankings when called twice with the same input.
    Determinism is a separate concern from correctness — it tests that
    the engine has no hidden state or randomness.
    """
    test_cases = build_test_suite()
    results: List[CaseResult] = []
    passed_count = 0

    for tc in test_cases:
        # Run guardrails and recommendation engine for each test case
        issues = run_guardrails(tc.profile, songs)
        recommendations = recommend_songs(tc.profile, songs, k=k)

        # Run every check function in this test case
        outcomes: List[CheckOutcome] = []
        test_passed = True   # will be set to False if any check fails
        for fn in tc.checks:
            ok, msg = fn(recommendations, issues)
            outcomes.append(CheckOutcome(check_name=fn.__name__, passed=ok, message=msg))
            if not ok:
                test_passed = False   # any single failure marks the whole case as FAIL

        # Extract top result metadata for the report
        top_score = recommendations[0][1] if recommendations else 0.0
        top_title = recommendations[0][0]["title"] if recommendations else "N/A"

        results.append(CaseResult(
            name=tc.name,
            description=tc.description,
            passed=test_passed,
            outcomes=outcomes,
            top_title=top_title,
            top_score=top_score,
            confidence=confidence_score(top_score),   # normalized 0–1 confidence
            guardrail_count=len(issues),
        ))
        if test_passed:
            passed_count += 1

    # --- T09: Determinism check ---
    # Runs the same profile twice and compares the ordered title lists.
    # The rule engine is deterministic by design (no randomness, no state),
    # so this should always pass — failure would indicate a bug in the engine.
    ref_profile = {"genre": "jazz", "mood": "relaxed", "target_energy": 0.35, "likes_acoustic": True}
    run1 = [r[0]["title"] for r in recommend_songs(ref_profile, songs, k=k)]   # first run
    run2 = [r[0]["title"] for r in recommend_songs(ref_profile, songs, k=k)]   # second run (same input)
    det_passed = run1 == run2   # must be identical in the same order
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
        top_score=0.0,      # not meaningful for a determinism check — always shown as 0
        confidence=0.0,
        guardrail_count=0,
    ))
    if det_passed:
        passed_count += 1

    total = len(results)   # T01–T09
    return EvaluationReport(
        total_tests=total,
        passed_tests=passed_count,
        failed_tests=total - passed_count,
        reliability_score=round(passed_count / total, 3),   # percentage as a decimal, 3dp
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
            # Show top result with score and normalized confidence percentage
            print(
                f"         Top result : {result.top_title!r} "
                f"(score={result.top_score:.2f}/6.5, confidence={result.confidence:.0%})"
            )
        if result.guardrail_count > 0:
            print(f"         Guardrails : {result.guardrail_count} issue(s) raised")
        for outcome in result.outcomes:
            tick = "ok" if outcome.passed else "!!"   # visual pass/fail indicator
            print(f"         [{tick}] {outcome.check_name}: {outcome.message}")

    print(f"\n{thin}")
    print(f"  Results           : {report.passed_tests}/{report.total_tests} passed")
    print(f"  Reliability score : {report.reliability_score:.0%}")

    # Letter grade thresholds — defined informally for this project
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
    songs = load_songs(_SONGS_PATH)         # load catalog from data/songs.csv
    print(f"Catalog loaded: {len(songs)} songs")
    report = run_test_suite(songs)          # run all 9 test cases
    print_report(report)                    # print formatted results to stdout


if __name__ == "__main__":
    main()
