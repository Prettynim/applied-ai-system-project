"""
Tests for src/guardrails.py

Covers:
- Clean profile produces no issues
- Energy out of range triggers ERROR
- Genre not in catalog triggers WARNING
- Mood not in catalog triggers WARNING
- High energy + acoustic triggers WARNING
- Thin genre coverage triggers INFO
- format_issues returns non-empty string when issues exist
"""

import pytest
from src.guardrails import (
    run_guardrails,
    format_issues,
    Severity,
)

# --- Minimal song catalog for testing ---

SONGS = [
    {
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "valence": 0.9, "acousticness": 0.2,
    },
    {
        "genre": "lofi", "mood": "focused", "energy": 0.35,
        "valence": 0.6, "acousticness": 0.8,
    },
    {
        "genre": "lofi", "mood": "chill", "energy": 0.3,
        "valence": 0.5, "acousticness": 0.75,
    },
    {
        "genre": "rock", "mood": "intense", "energy": 0.92,
        "valence": 0.7, "acousticness": 0.05,
    },
]


# ---------------------------------------------------------------------------
# Clean profiles
# ---------------------------------------------------------------------------

def test_clean_profile_no_warnings_or_errors():
    # Mini SONGS catalog has only 1 pop song, so THIN_GENRE_COVERAGE (INFO) fires.
    # A "clean" profile should have no WARNING or ERROR issues — INFO is acceptable.
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    blocking = [i for i in issues if i.severity in (Severity.WARNING, Severity.ERROR)]
    assert blocking == [], f"Unexpected WARNING/ERROR issues: {[i.code for i in blocking]}"


def test_lofi_acoustic_clean():
    profile = {"genre": "lofi", "mood": "focused", "target_energy": 0.35, "likes_acoustic": True}
    issues = run_guardrails(profile, SONGS)
    # likes_acoustic=True + energy=0.35 should NOT trigger HIGH_ENERGY_ACOUSTIC_CONFLICT
    codes = [i.code for i in issues]
    assert "HIGH_ENERGY_ACOUSTIC_CONFLICT" not in codes
    # No ERROR-level issues
    assert all(i.severity != Severity.ERROR for i in issues)


# ---------------------------------------------------------------------------
# Energy range
# ---------------------------------------------------------------------------

def test_energy_above_1_triggers_error():
    profile = {"genre": "pop", "mood": "happy", "target_energy": 1.5, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "ENERGY_OUT_OF_RANGE" in codes
    error_issues = [i for i in issues if i.code == "ENERGY_OUT_OF_RANGE"]
    assert error_issues[0].severity == Severity.ERROR


def test_energy_below_0_triggers_error():
    profile = {"genre": "pop", "mood": "happy", "target_energy": -0.1, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "ENERGY_OUT_OF_RANGE" in codes


def test_energy_exactly_0_is_valid():
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.0, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "ENERGY_OUT_OF_RANGE" not in codes


def test_energy_exactly_1_is_valid():
    profile = {"genre": "rock", "mood": "intense", "target_energy": 1.0, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "ENERGY_OUT_OF_RANGE" not in codes


# ---------------------------------------------------------------------------
# Genre catalog gap
# ---------------------------------------------------------------------------

def test_unknown_genre_triggers_warning():
    profile = {"genre": "reggae", "mood": "happy", "target_energy": 0.6, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "GENRE_NOT_IN_CATALOG" in codes
    gap_issue = next(i for i in issues if i.code == "GENRE_NOT_IN_CATALOG")
    assert gap_issue.severity == Severity.WARNING


def test_known_genre_no_genre_gap():
    profile = {"genre": "rock", "mood": "intense", "target_energy": 0.9, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "GENRE_NOT_IN_CATALOG" not in codes


# ---------------------------------------------------------------------------
# Mood catalog gap
# ---------------------------------------------------------------------------

def test_unknown_mood_triggers_warning():
    profile = {"genre": "lofi", "mood": "romantic", "target_energy": 0.4, "likes_acoustic": True}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "MOOD_NOT_IN_CATALOG" in codes
    mood_issue = next(i for i in issues if i.code == "MOOD_NOT_IN_CATALOG")
    assert mood_issue.severity == Severity.WARNING


def test_known_mood_no_mood_gap():
    profile = {"genre": "rock", "mood": "intense", "target_energy": 0.9, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "MOOD_NOT_IN_CATALOG" not in codes


# ---------------------------------------------------------------------------
# High-energy acoustic conflict
# ---------------------------------------------------------------------------

def test_high_energy_acoustic_conflict_fires():
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.9, "likes_acoustic": True}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "HIGH_ENERGY_ACOUSTIC_CONFLICT" in codes
    conflict = next(i for i in issues if i.code == "HIGH_ENERGY_ACOUSTIC_CONFLICT")
    assert conflict.severity == Severity.WARNING


def test_moderate_energy_acoustic_no_conflict():
    # energy=0.75 is the boundary; 0.74 should NOT trigger
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.74, "likes_acoustic": True}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "HIGH_ENERGY_ACOUSTIC_CONFLICT" not in codes


def test_high_energy_non_acoustic_no_conflict():
    profile = {"genre": "rock", "mood": "intense", "target_energy": 0.95, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "HIGH_ENERGY_ACOUSTIC_CONFLICT" not in codes


# ---------------------------------------------------------------------------
# Thin genre coverage
# ---------------------------------------------------------------------------

def test_thin_genre_coverage_info():
    # "pop" has only 1 song in SONGS
    profile = {"genre": "pop", "mood": "happy", "target_energy": 0.8, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "THIN_GENRE_COVERAGE" in codes
    info_issue = next(i for i in issues if i.code == "THIN_GENRE_COVERAGE")
    assert info_issue.severity == Severity.INFO


def test_multi_song_genre_no_thin_coverage():
    # "lofi" has 2 songs in SONGS — should not trigger THIN_GENRE_COVERAGE
    profile = {"genre": "lofi", "mood": "focused", "target_energy": 0.35, "likes_acoustic": True}
    issues = run_guardrails(profile, SONGS)
    codes = [i.code for i in issues]
    assert "THIN_GENRE_COVERAGE" not in codes


# ---------------------------------------------------------------------------
# format_issues
# ---------------------------------------------------------------------------

def test_format_issues_empty():
    output = format_issues([])
    assert "No issues" in output


def test_format_issues_non_empty():
    profile = {"genre": "reggae", "mood": "romantic", "target_energy": 0.9, "likes_acoustic": True}
    issues = run_guardrails(profile, SONGS)
    assert len(issues) > 0
    output = format_issues(issues)
    assert isinstance(output, str)
    assert len(output) > 0
    # Each issue code should appear in the formatted output
    for issue in issues:
        assert issue.code in output


def test_format_issues_severity_labels():
    profile = {"genre": "reggae", "mood": "happy", "target_energy": 0.5, "likes_acoustic": False}
    issues = run_guardrails(profile, SONGS)
    output = format_issues(issues)
    # WARNING issues should show "WARNING" label
    warning_issues = [i for i in issues if i.severity == Severity.WARNING]
    if warning_issues:
        assert "WARNING" in output
