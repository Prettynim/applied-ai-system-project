"""
AI-powered agentic layer for the Music Recommender.

Adds three capabilities on top of the rule-based engine:

1. Natural language -> UserProfile parsing
   Claude reads a free-text request ("something chill for studying") and extracts
   a structured profile (genre, mood, energy, acoustic preference).

2. Self-critique loop
   After the rule engine scores songs, Claude reviews the top results and flags
   mismatches, catalog gaps, or contradictions before the user sees them.

3. run_agentic_session()
   Full pipeline: NL input -> guardrails -> rule engine -> Claude critique -> output.

Usage:
    python -m src.agent
    python -m src.agent "I want something energetic for my morning run"
"""

import os
import sys
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import anthropic

try:
    from recommender import load_songs, recommend_songs
    from guardrails import run_guardrails, format_issues, GuardrailIssue
except ImportError:
    from src.recommender import load_songs, recommend_songs
    from src.guardrails import run_guardrails, format_issues, GuardrailIssue

# Resolve data path relative to this file so the module works from any directory
_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


# ---------------------------------------------------------------------------
# Pydantic schema for structured NL extraction
# ---------------------------------------------------------------------------

class ProfileExtraction(BaseModel):
    """Music taste profile extracted from natural language."""

    favorite_genre: str = Field(
        description=(
            "Best-matching music genre. Choose from: pop, rock, lofi, jazz, edm, hip-hop, "
            "r&b, classical, country, metal, funk, folk, ambient, synthwave, indie pop. "
            "Pick the closest match even if the user's phrasing differs."
        )
    )
    favorite_mood: str = Field(
        description=(
            "Best-matching mood. Choose from: happy, focused, relaxed, intense, euphoric, "
            "chill, melancholic, moody, energetic, romantic, dreamy, upbeat, peaceful, angry, "
            "nostalgic. Pick the closest match."
        )
    )
    target_energy: float = Field(
        description=(
            "Energy level from 0.0 (very calm, quiet, ambient) to 1.0 (very intense, loud, fast). "
            "Guidelines: sleep/meditation=0.1-0.2, study=0.25-0.45, casual=0.45-0.65, "
            "workout/dance=0.7-1.0."
        ),
        ge=0.0,
        le=1.0,
    )
    likes_acoustic: bool = Field(
        description=(
            "True if the user mentions acoustic, unplugged, live, folk, classical, natural-sounding, "
            "or organic instruments. False for electronic, produced, digital, or synth-heavy sound. "
            "Default False when not mentioned."
        )
    )
    extraction_confidence: float = Field(
        description=(
            "Confidence in this extraction from 0.0 to 1.0. "
            "Use 0.8-1.0 for clear, specific requests. "
            "Use 0.4-0.7 for vague or short requests. "
            "Use 0.1-0.4 for contradictory or ambiguous inputs."
        ),
        ge=0.0,
        le=1.0,
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description=(
            "Brief note on ambiguities, assumptions made, or missing info. "
            "Leave null if the request was clear."
        ),
    )


# ---------------------------------------------------------------------------
# Step 1: NL parsing
# ---------------------------------------------------------------------------

def parse_natural_language(user_text: str, client: anthropic.Anthropic) -> ProfileExtraction:
    """
    Uses Claude to parse a free-text music request into a structured ProfileExtraction.

    Returns a ProfileExtraction instance with genre, mood, energy, acoustic preference,
    a confidence score, and optional notes on ambiguities.
    """
    response = client.messages.parse(
        model="claude-opus-4-6",
        max_tokens=1024,
        system=(
            "You are a music taste analyst. Your only job is to extract structured music "
            "preferences from a user's natural language request. "
            "Be conservative with defaults: if something is not mentioned, use neutral values "
            "(target_energy around 0.5, likes_acoustic=False). "
            "Always pick a genre from the allowed list, choosing the closest match."
        ),
        messages=[{
            "role": "user",
            "content": f'Extract music preferences from this request:\n\n"{user_text}"',
        }],
        output_format=ProfileExtraction,
    )
    return response.parsed_output


# ---------------------------------------------------------------------------
# Step 2: Self-critique
# ---------------------------------------------------------------------------

def self_critique(
    user_text: str,
    profile: Dict,
    recommendations: List,
    guardrail_issues: List[GuardrailIssue],
    client: anthropic.Anthropic,
) -> str:
    """
    Uses Claude to critically review the rule-engine's recommendations.

    Streams the response and returns the complete critique text. Claude checks:
    - Whether results match the user's actual intent
    - Catalog limitations affecting quality
    - Overall confidence in the recommendations
    - One concrete improvement suggestion
    """
    rec_summary = "\n".join(
        f"  #{i + 1} {song['title']} by {song['artist']} "
        f"(genre={song['genre']}, mood={song['mood']}, "
        f"energy={song['energy']}, score={score:.2f}/6.5)"
        for i, (song, score, _) in enumerate(recommendations)
    )

    guardrail_summary = (
        "\n".join(
            f"  [{issue.severity.value.upper()}] {issue.code}: {issue.message}"
            for issue in guardrail_issues
        )
        if guardrail_issues
        else "  None"
    )

    prompt = f"""You are reviewing music recommendations produced by a rule-based scoring engine.

USER'S ORIGINAL REQUEST:
"{user_text}"

PARSED PROFILE:
- Genre: {profile['genre']}
- Mood: {profile['mood']}
- Energy: {profile['target_energy']}
- Likes acoustic: {profile['likes_acoustic']}

GUARDRAIL WARNINGS:
{guardrail_summary}

TOP {len(recommendations)} RECOMMENDATIONS (scored by rule engine, max 6.5 pts):
{rec_summary}

Review these recommendations critically. In 4-6 sentences, address:
1. Do the results match what the user actually asked for? Call out any obvious mismatches.
2. What catalog limitations (if any) are affecting quality?
3. Overall confidence that these recommendations are useful: low / medium / high — and why.
4. One concrete suggestion for improving the result."""

    parts: List[str] = []
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=512,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            parts.append(text)

    return "".join(parts)


# ---------------------------------------------------------------------------
# Full agentic session
# ---------------------------------------------------------------------------

def run_agentic_session(
    user_text: str,
    songs_path: str = _SONGS_PATH,
    k: int = 5,
) -> None:
    """
    Full pipeline from free-text input to Claude-reviewed recommendations.

    Steps:
      1. Claude parses the natural language request into a UserProfile.
      2. Guardrails check the profile for known failure modes.
      3. Rule-based engine scores all songs and ranks them.
      4. Claude self-critique reviews the results and flags issues.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[ERROR] ANTHROPIC_API_KEY environment variable is not set.")
        print("  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    client = anthropic.Anthropic(api_key=api_key)
    songs = load_songs(songs_path)

    width = 65
    sep = "=" * width
    thin = "-" * width

    print(f"\n{sep}")
    print("  AI MUSIC RECOMMENDER — Agentic Session")
    print(sep)
    print(f'\n  Request: "{user_text}"\n')

    # ------------------------------------------------------------------
    # Step 1: Parse NL -> profile
    # ------------------------------------------------------------------
    print("  [1/4] Parsing request with Claude...")
    try:
        extraction = parse_natural_language(user_text, client)
    except anthropic.APIError as e:
        print(f"  [ERROR] Claude API call failed: {e}")
        return

    profile = {
        "genre": extraction.favorite_genre,
        "mood": extraction.favorite_mood,
        "target_energy": extraction.target_energy,
        "likes_acoustic": extraction.likes_acoustic,
    }

    print("  Parsed profile:")
    print(f"    Genre   : {profile['genre']}   Mood : {profile['mood']}")
    print(f"    Energy  : {profile['target_energy']:.2f}   Acoustic : {profile['likes_acoustic']}")
    print(f"    Confidence : {extraction.extraction_confidence:.0%}")
    if extraction.extraction_notes:
        print(f"    Notes   : {extraction.extraction_notes}")

    # ------------------------------------------------------------------
    # Step 2: Guardrails
    # ------------------------------------------------------------------
    print(f"\n  [2/4] Running guardrails...")
    issues = run_guardrails(profile, songs)
    if issues:
        print(format_issues(issues))
    else:
        print("  No issues detected.")

    # ------------------------------------------------------------------
    # Step 3: Rule-based recommendations
    # ------------------------------------------------------------------
    print(f"\n  [3/4] Scoring {len(songs)} songs with rule engine...")
    recommendations = recommend_songs(profile, songs, k=k)

    print(f"\n  Top {k} Recommendations:")
    for rank, (song, score, explanation) in enumerate(recommendations, 1):
        pct = score / 6.5
        bar = "#" * int(pct * 10) + "-" * (10 - int(pct * 10))
        print(f"\n    #{rank}  {song['title']} by {song['artist']}")
        print(f"         Score  : {score:.2f}/6.5  [{bar}] {pct:.0%} match")
        print(f"         Genre  : {song['genre']} | Mood: {song['mood']} | Energy: {song['energy']}")
        print("         Why    :")
        for reason in explanation.split("; "):
            print(f"           - {reason}")

    # ------------------------------------------------------------------
    # Step 4: Claude self-critique
    # ------------------------------------------------------------------
    print(f"\n  [4/4] Claude self-critique...")
    print(f"  {thin}")
    try:
        critique = self_critique(user_text, profile, recommendations, issues, client)
        for line in critique.strip().splitlines():
            print(f"  {line}")
    except anthropic.APIError as e:
        print(f"  [ERROR] Critique call failed: {e}")
    print(f"  {thin}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        # Request passed as a command-line argument
        user_text = " ".join(sys.argv[1:])
        run_agentic_session(user_text)
    else:
        # Interactive mode
        print("\nAI Music Recommender — Agentic Mode")
        print("Type your music request in plain English. Press Ctrl+C to quit.\n")
        try:
            while True:
                user_text = input("Your request: ").strip()
                if not user_text:
                    continue
                run_agentic_session(user_text)
                print()
        except KeyboardInterrupt:
            print("\nGoodbye.")


if __name__ == "__main__":
    main()
