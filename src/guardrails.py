"""
Guardrails for the Music Recommender.

Checks user profiles for known failure modes before recommendations are made:
- Energy value out of valid range
- Genre not represented in catalog (genre gap)
- Mood not represented in catalog (mood gap)
- Conflicting preferences (high energy + acoustic)
- Thin catalog coverage for a genre
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Set


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class GuardrailIssue:
    severity: Severity
    code: str
    message: str
    suggestion: str


def run_guardrails(profile: Dict, songs: List[Dict]) -> List[GuardrailIssue]:
    """
    Run all guardrail checks on a user profile against the song catalog.
    Returns a list of issues found (empty list = profile is clean).
    """
    issues: List[GuardrailIssue] = []

    known_genres: Set[str] = {s["genre"] for s in songs}
    known_moods: Set[str] = {s["mood"] for s in songs}

    # --- 1. Energy range ---
    energy = profile.get("target_energy", 0.5)
    if not isinstance(energy, (int, float)) or not 0.0 <= float(energy) <= 1.0:
        issues.append(GuardrailIssue(
            severity=Severity.ERROR,
            code="ENERGY_OUT_OF_RANGE",
            message=f"target_energy={energy} is outside the valid range [0.0, 1.0].",
            suggestion="Use 0.0 for very calm / ambient, 1.0 for maximum intensity.",
        ))

    # --- 2. Genre catalog gap ---
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

    # --- 3. Mood catalog gap ---
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

    # --- 4. High-energy acoustic contradiction ---
    likes_acoustic = profile.get("likes_acoustic", False)
    energy_val = float(energy) if isinstance(energy, (int, float)) else 0.5
    if likes_acoustic and energy_val > 0.75:
        conflicting_songs = sum(
            1 for s in songs
            if s["acousticness"] > 0.6 and s["energy"] > 0.75
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

    # --- 5. Thin genre coverage (info only) ---
    genre_count = sum(1 for s in songs if s["genre"] == genre)
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

    return issues


def format_issues(issues: List[GuardrailIssue]) -> str:
    """Returns a human-readable string summarizing all guardrail issues."""
    if not issues:
        return "  No issues detected."

    icons = {Severity.INFO: "[i]", Severity.WARNING: "[!]", Severity.ERROR: "[x]"}
    lines = []
    for issue in issues:
        icon = icons[issue.severity]
        lines.append(f"  {icon} {issue.severity.value.upper()} — {issue.code}")
        lines.append(f"      {issue.message}")
        lines.append(f"      -> {issue.suggestion}")
    return "\n".join(lines)
