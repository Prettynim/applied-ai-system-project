"""
AI-powered agentic layer for the Music Recommender.

Pipeline (4 steps):
  1. RAG-grounded NL parsing
     Claude reads a free-text request and maps it to a structured UserProfile.
     A catalog genre overview is RETRIEVED from the knowledge base and injected
     into the system prompt so Claude's genre selection is grounded in what
     actually exists in the catalog.

  2. Guardrails
     The parsed profile is checked for known failure modes before any
     recommendation is computed (energy range, genre gaps, contradictions).

  3. Rule-based recommendation engine
     The existing weighted scoring engine ranks all 18 songs.

  4. RAG-grounded self-critique
     The genre profile for the TOP recommended song is RETRIEVED from the
     knowledge base and given to Claude. Claude compares actual results
     against expected genre characteristics and writes a structured critique.

All steps are logged to logs/sessions.log via the session logger.

Usage:
    python -m src.agent
    python -m src.agent "I want something energetic for my morning run"
"""

import os
import sys
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
import anthropic

try:
    from recommender import load_songs, recommend_songs
    from guardrails import run_guardrails, format_issues, GuardrailIssue, Severity
    from rag import (
        load_knowledge_base,
        format_catalog_overview,
        format_genre_context,
        find_closest_catalog_genre,
    )
    from logger import get_logger, new_session_id
except ImportError:
    from src.recommender import load_songs, recommend_songs
    from src.guardrails import run_guardrails, format_issues, GuardrailIssue, Severity
    from src.rag import (
        load_knowledge_base,
        format_catalog_overview,
        format_genre_context,
        find_closest_catalog_genre,
    )
    from src.logger import get_logger, new_session_id

_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


# ---------------------------------------------------------------------------
# Pydantic schema for structured NL extraction
# ---------------------------------------------------------------------------

class ProfileExtraction(BaseModel):
    """Music taste profile extracted from natural language."""

    favorite_genre: str = Field(
        description=(
            "Best-matching genre from the AVAILABLE CATALOG GENRES listed in context. "
            "You MUST pick from that list. If the user's genre is not available, "
            "choose the closest catalog match based on the energy and mood profiles shown."
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
            "Energy level 0.0 (very calm/ambient) to 1.0 (very intense/loud). "
            "Use the energy ranges in the genre overview as a guide. "
            "Guidelines: sleep=0.1-0.2, study=0.25-0.45, casual=0.45-0.65, workout/dance=0.7-1.0."
        ),
        ge=0.0,
        le=1.0,
    )
    likes_acoustic: bool = Field(
        description=(
            "True if the user mentions acoustic, unplugged, live, folk, classical, "
            "natural-sounding, or organic instruments. "
            "False for electronic, produced, digital, or synth-heavy. Default False."
        )
    )
    extraction_confidence: float = Field(
        description=(
            "Confidence in this extraction 0.0-1.0. "
            "0.8-1.0 = clear specific request. "
            "0.4-0.7 = vague or short. "
            "0.1-0.4 = contradictory or ambiguous."
        ),
        ge=0.0,
        le=1.0,
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description=(
            "Brief note on ambiguities, assumptions, or genre remapping performed. "
            "Required if requested genre is not in catalog. Null if request was clear."
        ),
    )


# ---------------------------------------------------------------------------
# Step 1: RAG-grounded NL parsing
# ---------------------------------------------------------------------------

def parse_natural_language(
    user_text: str,
    client: anthropic.Anthropic,
    catalog_overview: str,
    log: logging.Logger,
) -> ProfileExtraction:
    """
    Parse a free-text music request into a structured UserProfile.

    RAG integration: `catalog_overview` (retrieved from knowledge/genres.json)
    is injected into the system prompt so Claude's genre selection is anchored
    to genres that actually exist in the catalog, with factual energy ranges.
    """
    log.info("Step 1: Calling Claude API for NL parsing (RAG-grounded)")
    log.debug("User text: %s", user_text)

    system_prompt = (
        "You are a music taste analyst. Extract structured music preferences "
        "from a user's natural language request.\n\n"
        + catalog_overview
        + "\n\n"
        "Rules:\n"
        "- favorite_genre MUST be one of the genres listed above.\n"
        "- If the user requests a genre not in the catalog, pick the closest "
        "match based on the energy and mood profiles above and note the remapping.\n"
        "- Be conservative with defaults: if energy is not mentioned, use the "
        "midpoint of the closest genre's typical range.\n"
        "- likes_acoustic defaults to False unless explicitly implied."
    )

    try:
        response = client.messages.parse(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f'Extract music preferences from this request:\n\n"{user_text}"',
            }],
            output_format=ProfileExtraction,
        )
        extraction = response.parsed_output
        log.info(
            "Parsed: genre=%s mood=%s energy=%.2f acoustic=%s confidence=%.0f%%",
            extraction.favorite_genre,
            extraction.favorite_mood,
            extraction.target_energy,
            extraction.likes_acoustic,
            extraction.extraction_confidence * 100,
        )
        if extraction.extraction_notes:
            log.info("Extraction notes: %s", extraction.extraction_notes)
        return extraction

    except anthropic.APIError as e:
        log.error("Claude API error during NL parsing: %s", e)
        raise


# ---------------------------------------------------------------------------
# Step 4: RAG-grounded self-critique
# ---------------------------------------------------------------------------

def self_critique(
    user_text: str,
    profile: Dict,
    recommendations: List,
    guardrail_issues: List[GuardrailIssue],
    genre_context: str,
    client: anthropic.Anthropic,
    log: logging.Logger,
) -> str:
    """
    Stream a Claude critique of the rule engine's recommendations.

    RAG integration: `genre_context` (retrieved for the top recommended song's
    genre) is injected into the prompt. Claude compares actual results against
    the retrieved expected characteristics (energy range, typical moods) and
    flags mismatches that the rule engine cannot detect on its own.
    """
    log.info("Step 4: Calling Claude API for self-critique (RAG-grounded)")

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
Use the retrieved genre knowledge below to ground your critique in factual data.

{genre_context}

USER'S ORIGINAL REQUEST:
"{user_text}"

PARSED PROFILE:
- Genre: {profile['genre']}
- Mood: {profile['mood']}
- Energy: {profile['target_energy']}
- Likes acoustic: {profile['likes_acoustic']}

GUARDRAIL WARNINGS:
{guardrail_summary}

TOP {len(recommendations)} RECOMMENDATIONS (rule engine, max 6.5 pts):
{rec_summary}

Using the retrieved genre context above, critically review these recommendations in 4-6 sentences:
1. Do the results match the user's request? Compare actual song energies to the retrieved typical range.
2. Does the top result's genre and mood align with what the user described?
3. What catalog limitations or scoring biases affected the results?
4. Overall confidence: low / medium / high — and one concrete suggestion to improve."""

    parts: List[str] = []
    try:
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=512,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                parts.append(text)
        critique = "".join(parts)
        log.info("Self-critique complete (%d chars)", len(critique))
        log.debug("Critique text: %s", critique[:300])
        return critique

    except anthropic.APIError as e:
        log.error("Claude API error during self-critique: %s", e)
        raise


# ---------------------------------------------------------------------------
# Full agentic session
# ---------------------------------------------------------------------------

def run_agentic_session(
    user_text: str,
    songs_path: str = _SONGS_PATH,
    k: int = 5,
) -> None:
    """
    Full pipeline: free-text input -> RAG-grounded NL parse -> guardrails
    -> rule engine -> RAG-grounded Claude critique -> printed output.

    Logs every step to logs/sessions.log.
    """
    session_id = new_session_id()
    log = get_logger(session_id)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY environment variable is not set")
        print("\n[ERROR] ANTHROPIC_API_KEY is not set.")
        print("  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    log.info("=== Session started | request: %s", user_text)

    try:
        songs = load_songs(songs_path)
    except FileNotFoundError:
        log.error("Songs catalog not found at: %s", songs_path)
        print(f"\n[ERROR] Cannot find songs catalog: {songs_path}")
        return

    log.info("Catalog loaded: %d songs", len(songs))

    kb = load_knowledge_base()
    catalog_genres = {s["genre"] for s in songs}

    # RAG retrieval for Step 1
    catalog_overview = format_catalog_overview(catalog_genres, kb)
    log.debug("RAG: catalog overview retrieved (%d genres)", len(catalog_genres))

    client = anthropic.Anthropic(api_key=api_key)

    width = 65
    sep = "=" * width
    thin = "-" * width

    print(f"\n{sep}")
    print(f"  AI MUSIC RECOMMENDER — Agentic Session [{session_id}]")
    print(sep)
    print(f'\n  Request: "{user_text}"\n')

    # ------------------------------------------------------------------
    # Step 1: RAG-grounded NL parsing
    # ------------------------------------------------------------------
    print("  [1/4] Parsing request with Claude (RAG-grounded)...")
    try:
        extraction = parse_natural_language(user_text, client, catalog_overview, log)
    except anthropic.APIError:
        print("  [ERROR] Claude API call failed. Check logs/sessions.log for details.")
        log.info("=== Session ended with error (Step 1)")
        return

    profile = {
        "genre": extraction.favorite_genre,
        "mood": extraction.favorite_mood,
        "target_energy": extraction.target_energy,
        "likes_acoustic": extraction.likes_acoustic,
    }

    print("  Parsed profile:")
    print(f"    Genre      : {profile['genre']}   Mood : {profile['mood']}")
    print(f"    Energy     : {profile['target_energy']:.2f}   Acoustic : {profile['likes_acoustic']}")
    print(f"    Confidence : {extraction.extraction_confidence:.0%}")
    if extraction.extraction_notes:
        print(f"    Notes      : {extraction.extraction_notes}")

    # Warn if confidence is low
    if extraction.extraction_confidence < 0.5:
        log.warning(
            "Low extraction confidence (%.0f%%) — profile may not reflect user intent",
            extraction.extraction_confidence * 100,
        )

    # ------------------------------------------------------------------
    # Step 2: Guardrails
    # ------------------------------------------------------------------
    print(f"\n  [2/4] Running guardrails...")
    issues = run_guardrails(profile, songs)

    for issue in issues:
        if issue.severity == Severity.ERROR:
            log.error("Guardrail ERROR — %s: %s", issue.code, issue.message)
        elif issue.severity == Severity.WARNING:
            log.warning("Guardrail WARNING — %s: %s", issue.code, issue.message)
        else:
            log.info("Guardrail INFO — %s: %s", issue.code, issue.message)

    # RAG fallback: if genre gap detected, find closest catalog match
    genre_gap = next((i for i in issues if i.code == "GENRE_NOT_IN_CATALOG"), None)
    if genre_gap:
        closest = find_closest_catalog_genre(profile["genre"], catalog_genres, kb)
        if closest:
            log.info("RAG fallback: '%s' not in catalog, closest match is '%s'", profile["genre"], closest)
            print(f"  [RAG] Genre '{profile['genre']}' not in catalog.")
            print(f"        Knowledge base suggests closest match: '{closest}'")

    if issues:
        print(format_issues(issues))
    else:
        print("  No issues detected.")

    # ------------------------------------------------------------------
    # Step 3: Rule-based recommendations
    # ------------------------------------------------------------------
    print(f"\n  [3/4] Scoring {len(songs)} songs with rule engine...")
    recommendations = recommend_songs(profile, songs, k=k)
    log.info(
        "Recommendations: top result='%s' score=%.2f",
        recommendations[0][0]["title"] if recommendations else "N/A",
        recommendations[0][1] if recommendations else 0.0,
    )

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
    # Step 4: RAG-grounded self-critique
    # ------------------------------------------------------------------
    print(f"\n  [4/4] Claude self-critique (RAG-grounded)...")

    # Retrieve genre context for the top result's genre
    top_genre = recommendations[0][0]["genre"] if recommendations else profile["genre"]
    genre_context = format_genre_context(top_genre, kb)
    log.debug("RAG: retrieved context for genre '%s'", top_genre)

    print(f"  {thin}")
    try:
        critique = self_critique(
            user_text, profile, recommendations, issues, genre_context, client, log
        )
        for line in critique.strip().splitlines():
            print(f"  {line}")
    except anthropic.APIError:
        print("  [ERROR] Critique call failed. Check logs/sessions.log.")
        log.info("=== Session ended with error (Step 4)")
    else:
        log.info("=== Session completed successfully")
    print(f"  {thin}")

    print(f"\n  Log written to: logs/sessions.log  [session={session_id}]")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
        run_agentic_session(user_text)
    else:
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
