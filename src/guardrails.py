"""
Guardrails for the Music Recommender.

Checks user profiles for known failure modes before recommendations are made:
- Energy value out of valid range
- Genre not represented in catalog (genre gap)
- Mood not represented in catalog (mood gap)
- Conflicting preferences (high energy + acoustic)
- Thin catalog coverage for a genre

All checks run on every request, before the scoring engine is invoked.
Results are returned as a list of GuardrailIssue objects so the caller
(agent.py) can log, display, and act on them independently.
"""

from dataclasses import dataclass   # lightweight data class — no custom __init__ needed
from enum import Enum               # typed severity levels — prevents typos like "warning" vs "Warning"
from typing import List, Dict, Set  # type hints for function signatures


class Severity(Enum):
    """
    Three-level severity scale for guardrail issues.
    INFO    — informational; does not block recommendations (e.g., thin catalog coverage)
    WARNING — notable issue that degrades results but doesn't prevent them (e.g., genre gap)
    ERROR   — invalid input that the scoring engine cannot handle correctly (e.g., energy > 1.0)
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class GuardrailIssue:
    """
    A single guardrail finding. All four fields are always populated.
    `suggestion` is surfaced to the user; `message` explains the problem.
    """
    severity: Severity   # determines icon displayed and log level used
    code: str            # machine-readable identifier (e.g., "GENRE_NOT_IN_CATALOG")
    message: str         # human-readable description of what was detected
    suggestion: str      # actionable next step for the user


def run_guardrails(profile: Dict, songs: List[Dict]) -> List[GuardrailIssue]:
    """
    Run all guardrail checks on a user profile against the song catalog.
    Returns a list of issues found (empty list = profile is clean).

    Checks run in order: energy → genre → mood → acoustic conflict → coverage.
    All checks always execute — no early exit — so the user sees every issue at once.
    """
    issues: List[GuardrailIssue] = []

    # Build sets of valid genres and moods from the actual catalog at runtime.
    # Using sets (not hardcoded lists) means checks stay correct if songs.csv changes.
    known_genres: Set[str] = {s["genre"] for s in songs}
    known_moods: Set[str] = {s["mood"] for s in songs}

    # --- Check 1: Energy range validation ---
    # The scoring engine computes energy closeness as `1.0 - abs(target - song_energy)`.
    # If target_energy is outside [0, 1], the closeness formula produces values > 1.0,
    # which inflates scores and breaks the 6.5-point maximum. This is an ERROR.
    energy = profile.get("target_energy", 0.5)
    if not isinstance(energy, (int, float)) or not 0.0 <= float(energy) <= 1.0:
        issues.append(GuardrailIssue(
            severity=Severity.ERROR,
            code="ENERGY_OUT_OF_RANGE",
            message=f"target_energy={energy} is outside the valid range [0.0, 1.0].",
            suggestion="Use 0.0 for very calm / ambient, 1.0 for maximum intensity.",
        ))

    # --- Check 2: Genre catalog gap ---
    # If the requested genre isn't in the catalog, no song can earn the +2.0 genre bonus.
    # This is a WARNING (not ERROR) because the system can still return results via
    # mood and energy matching — they just won't have a genre match.
    genre = profile.get("genre", "")
    if genre not in known_genres:
        issues.append(GuardrailIssue(
            severity=Severity.WARNING,
            code="GENRE_NOT_IN_CATALOG",
            message=(
                f"Genre '{genre}' is not in the catalog. "
                f"The genre bonus (+2.0 pts) cannot be earned."
            ),
            suggestion=(
                f"Available genres: {', '.join(sorted(known_genres))}. "
                f"The system will fall back to mood and energy matching only."
            ),
        ))

    # --- Check 3: Mood catalog gap ---
    # Same logic as genre gap: if the mood isn't in the catalog, the +1.5 mood bonus
    # cannot be earned. System still returns results using genre and energy signals.
    mood = profile.get("mood", "")
    if mood not in known_moods:
        issues.append(GuardrailIssue(
            severity=Severity.WARNING,
            code="MOOD_NOT_IN_CATALOG",
            message=(
                f"Mood '{mood}' is not in the catalog. "
                f"The mood bonus (+1.5 pts) cannot be earned."
            ),
            suggestion=(
                f"Available moods: {', '.join(sorted(known_moods))}. "
                f"The system will rely on genre and energy signals only."
            ),
        ))

    # --- Check 4: High-energy + acoustic contradiction ---
    # High-energy acoustic songs are rare in the catalog — most acoustic songs are quiet.
    # When a user asks for high energy AND acoustic, the genre bonus (+2.0) and acoustic bonus (+1.0)
    # can push a quiet folk/classical song to #1 even though its energy contradicts the target.
    # Threshold of 0.75 chosen because it's above the "workout" energy range where
    # the contradiction becomes practically significant.
    likes_acoustic = profile.get("likes_acoustic", False)
    energy_val = float(energy) if isinstance(energy, (int, float)) else 0.5
    if likes_acoustic and energy_val > 0.75:
        # Count how many catalog songs actually satisfy both constraints simultaneously
        conflicting_songs = sum(
            1 for s in songs
            if s["acousticness"] > 0.6 and s["energy"] > 0.75   # acoustic AND high-energy
        )
        issues.append(GuardrailIssue(
            severity=Severity.WARNING,
            code="HIGH_ENERGY_ACOUSTIC_CONFLICT",
            message=(
                f"Contradiction: target_energy={energy_val:.2f} (high) with likes_acoustic=True. "
                f"Only {conflicting_songs} song(s) in catalog satisfy both constraints."
            ),
            suggestion=(
                "The genre bonus (+2.0) may override energy proximity. "
                "A quiet folk or classical song could rank #1 due to genre+acoustic bonuses "
                "even though its energy is far below your target."
            ),
        ))

    # --- Check 5: Thin genre coverage (informational only) ---
    # When a genre has exactly 1 song in the catalog, the top result is predetermined —
    # and positions #2–#5 come entirely from other genres with no genre bonus.
    # This is INFO (not WARNING) because it's a catalog limitation, not a profile problem.
    genre_count = sum(1 for s in songs if s["genre"] == genre)   # count matching songs
    if genre_count == 1:
        issues.append(GuardrailIssue(
            severity=Severity.INFO,
            code="THIN_GENRE_COVERAGE",
            message=(
                f"Only 1 song in the catalog matches genre '{genre}'. "
                f"Recommendations beyond #1 will come from non-genre signals."
            ),
            suggestion="Results may lack variety. Consider a related genre for broader options.",
        ))

    return issues   # empty list signals a clean profile


def format_issues(issues: List[GuardrailIssue]) -> str:
    """
    Returns a human-readable string summarizing all guardrail issues.
    Each issue shows an icon, severity label, code, message, and suggestion.
    Used by the agent to print issues to the terminal.
    """
    if not issues:
        return "  No issues detected."   # clean profile — no output needed

    # Map severity to a visual icon for quick scanning in the terminal
    icons = {Severity.INFO: "[i]", Severity.WARNING: "[!]", Severity.ERROR: "[x]"}
    lines = []
    for issue in issues:
        icon = icons[issue.severity]
        lines.append(f"  {icon} {issue.severity.value.upper()} - {issue.code}")
        lines.append(f"      {issue.message}")
        lines.append(f"      -> {issue.suggestion}")
    return "\n".join(lines)   # single string with embedded newlines for print()
