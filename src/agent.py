"""
AI-powered agentic layer for the Music Recommender (VibeFinder).

Pipeline (4 steps):
  1. Dual-RAG-grounded NL parsing
     Claude reads a free-text request and maps it to a structured UserProfile
     (Pydantic, structured output via messages.parse). A catalog genre overview
     is RETRIEVED from knowledge/genres.json and injected into the system prompt
     with few-shot calibration examples so Claude's genre mapping is grounded
     in what actually exists in the catalog.

  2. Guardrails
     The parsed profile is checked for 5 known failure modes before any
     recommendation is computed (ENERGY_OUT_OF_RANGE, GENRE_NOT_IN_CATALOG,
     MOOD_NOT_IN_CATALOG, HIGH_ENERGY_ACOUSTIC_CONFLICT, THIN_GENRE_COVERAGE).

  3. Agentic tool call — get_song_recommendations
     Claude receives the profile + dual-RAG context (genre + mood knowledge)
     and issues a structured tool call. The tool call parameters are logged
     as an observable intermediate step. The rule-based scoring engine executes
     the call deterministically and returns the top-k ranked songs.

  4. Streaming RAG-grounded critique
     The tool result is fed back to Claude via a second API call (messages.stream).
     Claude writes a streaming critique grounded in both the genre profile
     (knowledge/genres.json) and mood profile (knowledge/moods.json),
     identifying any energy/valence mismatches against expected ranges.

All steps are logged to logs/sessions.log via the session logger.

Usage:
    python -m src.agent
    python -m src.agent "I want something energetic for my morning run"
"""

import os          # os.path for data file paths; os.environ for API key
import sys         # sys.argv to accept optional CLI argument
import logging     # standard library logger, wrapped by src/logger.py
from typing import Optional, List, Dict   # type hints for all function signatures
from pydantic import BaseModel, Field     # structured output schema + field-level validators
import anthropic   # Anthropic Python SDK — messages.parse, messages.create, messages.stream

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Dual import path: supports both `python -m src.agent` (package mode)
# and direct execution from the src/ directory (`python agent.py`)
try:
    from recommender import load_songs, recommend_songs          # rule-based scoring engine
    from guardrails import run_guardrails, format_issues, GuardrailIssue, Severity  # pre-flight checks
    from rag import (
        load_knowledge_base,         # loads knowledge/genres.json into a dict
        format_catalog_overview,     # builds genre table for NL parsing prompt (RAG retrieval ①)
        format_genre_context,        # formats single genre profile for critique prompt (RAG retrieval ②)
        find_closest_catalog_genre,  # RAG fallback: maps off-catalog genre to nearest catalog genre
        load_mood_knowledge_base,    # loads knowledge/moods.json into a dict
        format_mood_context,         # formats single mood profile for critique prompt (RAG retrieval ③)
    )
    from logger import get_logger, new_session_id  # session-scoped logger + UUID generator
except ImportError:
    # Package-relative imports used when running as `python -m src.agent`
    from src.recommender import load_songs, recommend_songs
    from src.guardrails import run_guardrails, format_issues, GuardrailIssue, Severity
    from src.rag import (
        load_knowledge_base,
        format_catalog_overview,
        format_genre_context,
        find_closest_catalog_genre,
        load_mood_knowledge_base,
        format_mood_context,
    )
    from src.logger import get_logger, new_session_id

# Absolute path to the song catalog CSV, resolved relative to this file's location
_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


# ---------------------------------------------------------------------------
# Pydantic schema for structured NL extraction
# ---------------------------------------------------------------------------

class ProfileExtraction(BaseModel):
    """
    Schema that Claude must populate when called with messages.parse().
    Pydantic enforces field types and ge/le constraints before the value
    ever reaches the rest of the pipeline.
    """

    favorite_genre: str = Field(
        description=(
            "Best-matching genre from the AVAILABLE CATALOG GENRES listed in context. "
            "You MUST pick from that list. If the user's genre is not available, "
            "choose the closest catalog match based on the energy and mood profiles shown."
        )
        # No enum constraint here — Claude picks from the RAG-injected list at runtime,
        # so the valid set varies with the catalog rather than being hardcoded
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
        ge=0.0,   # Pydantic min validator — prevents energy < 0 reaching the scoring engine
        le=1.0,   # Pydantic max validator — prevents energy > 1 reaching the scoring engine
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
        ge=0.0,  # confidence cannot be negative
        le=1.0,  # confidence cannot exceed 100%
    )
    extraction_notes: Optional[str] = Field(
        default=None,   # null for clear requests — only populated when assumptions were made
        description=(
            "Brief note on ambiguities, assumptions, or genre remapping performed. "
            "Required if requested genre is not in catalog. Null if request was clear."
        ),
    )


# ---------------------------------------------------------------------------
# Few-shot examples for NL parsing specialization (Step 1)
# ---------------------------------------------------------------------------

# These four examples calibrate Claude's confidence scoring and genre-remapping behavior.
# Without them, Claude may assign 0.8+ confidence to vague inputs or silently remap
# off-catalog genres without noting it. The examples anchor the expected output format.
_FEW_SHOT_EXAMPLES = """
FEW-SHOT CALIBRATION EXAMPLES (follow this confidence and remapping pattern):

  Request: "I want lofi beats for deep focus, acoustic vibes"
  → genre=lofi  mood=focused  energy=0.35  acoustic=True  confidence=0.90  notes=null

  Request: "Something upbeat and danceable, I'm going out tonight"
  → genre=edm   mood=euphoric  energy=0.90  acoustic=False  confidence=0.85  notes=null

  Request: "Something nice for the background, not sure really"
  → genre=ambient  mood=relaxed  energy=0.30  acoustic=False  confidence=0.35
    notes="Very vague request. Defaulted to ambient/relaxed as most neutral background profile."

  Request: "I love reggae, upbeat feel-good vibes"
  → genre=r&b  mood=happy  energy=0.65  acoustic=False  confidence=0.62
    notes="Reggae not in catalog. Mapped to r&b based on similar groove and positive energy profile."
"""


# ---------------------------------------------------------------------------
# Step 1: RAG-grounded NL parsing
# ---------------------------------------------------------------------------

def parse_natural_language(
    user_text: str,
    client: anthropic.Anthropic,
    catalog_overview: str,   # pre-retrieved genre table from knowledge/genres.json
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

    # Build system prompt by concatenating: role instructions + RAG context + few-shot examples
    # Order matters: RAG context comes before the few-shot examples so Claude sees
    # what genres are available before it sees the calibration examples
    system_prompt = (
        "You are a music taste analyst. Extract structured music preferences "
        "from a user's natural language request.\n\n"
        + catalog_overview          # RAG retrieval ①: genre table injected here
        + "\n\n"
        "Rules:\n"
        "- favorite_genre MUST be one of the genres listed above.\n"
        "- If the user requests a genre not in the catalog, pick the closest "
        "match based on the energy and mood profiles above and note the remapping.\n"
        "- Be conservative with defaults: if energy is not mentioned, use the "
        "midpoint of the closest genre's typical range.\n"
        "- likes_acoustic defaults to False unless explicitly implied."
        + "\n\n"
        + _FEW_SHOT_EXAMPLES        # specialization: 4 calibration examples anchoring confidence scores
    )

    try:
        # messages.parse() returns a strongly-typed ProfileExtraction object.
        # The SDK validates that Claude's JSON output matches the Pydantic schema
        # before returning — no manual parsing needed.
        response = client.messages.parse(
            model="claude-opus-4-6",   # latest Claude model — best instruction-following
            max_tokens=1024,           # generous budget: schema + notes fit in ~200 tokens; 1024 is safe
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f'Extract music preferences from this request:\n\n"{user_text}"',
            }],
            output_format=ProfileExtraction,   # structured output: SDK enforces the Pydantic schema
        )
        extraction = response.parsed_output   # type: ProfileExtraction — fully validated object

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
        raise   # propagate to caller — session handler catches and exits gracefully


# ---------------------------------------------------------------------------
# Steps 3+4: Agentic tool-use — Claude calls the recommendation engine
# ---------------------------------------------------------------------------

def run_agentic_recommendation_and_critique(
    user_text: str,
    profile: Dict,
    guardrail_issues: List[GuardrailIssue],
    genre_context: str,   # RAG retrieval ②: genre profile from knowledge/genres.json
    mood_context: str,    # RAG retrieval ③: mood profile from knowledge/moods.json
    songs: List[Dict],
    client: anthropic.Anthropic,
    log: logging.Logger,
    k: int = 5,           # number of recommendations to return
) -> tuple:
    """
    Agentic Steps 3+4: Claude calls get_song_recommendations as a tool,
    receives the ranked results, then writes a critique grounded in
    dual-RAG context (genre knowledge + mood knowledge).

    Observable intermediate step: the tool call parameters are logged
    and printed so the decision-making chain is fully visible.

    Returns: (recommendations, critique, tool_call_summary_str)
    """

    # Tool definition sent to the Anthropic API — describes the function Claude can call.
    # The input_schema follows JSON Schema format; the API validates Claude's tool call
    # against this schema before returning it to us.
    tool_def = {
        "name": "get_song_recommendations",
        "description": (
            "Run the rule-based scoring engine against the 18-song catalog. "
            "Scores each song on: genre match (+2.0), mood match (+1.5), "
            "energy closeness (+1.5), acoustic bonus (+1.0), valence closeness (+0.5). "
            "Max score is 6.5. Returns the top-k ranked songs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre":          {"type": "string",  "description": "Genre from catalog"},
                "mood":           {"type": "string",  "description": "Mood from catalog"},
                "target_energy":  {"type": "number",  "description": "Energy level 0.0–1.0"},
                "likes_acoustic": {"type": "boolean", "description": "Acoustic preference"},
                "k":              {"type": "integer", "description": "Number of results"},
            },
            "required": ["genre", "mood", "target_energy", "likes_acoustic"],  # k is optional (defaults to 5)
        },
    }

    # Summarize any guardrail issues so Claude sees them when writing the critique
    guardrail_summary = (
        "\n".join(
            f"  [{i.severity.value.upper()}] {i.code}: {i.message}"
            for i in guardrail_issues
        ) if guardrail_issues else "  None"
    )

    # Initial prompt includes: role + dual-RAG context + profile + instructions for both turns.
    # Providing both genre and mood context here means Claude already has the factual data
    # it needs for the critique before any tool call happens.
    initial_prompt = (
        "You are a music recommendation assistant with access to a scoring engine.\n\n"
        f"{genre_context}\n\n"    # RAG retrieval ②: genre data injected
        f"{mood_context}\n\n"     # RAG retrieval ③: mood data injected
        f'USER REQUEST: "{user_text}"\n'
        f"PARSED PROFILE: genre={profile['genre']}, mood={profile['mood']}, "
        f"energy={profile['target_energy']}, acoustic={profile['likes_acoustic']}\n"
        f"GUARDRAIL RESULTS:\n{guardrail_summary}\n\n"
        f"Step 1: Call get_song_recommendations with the profile above (k={k}).\n"
        "Step 2: After receiving results, write a 4–6 sentence critique using the "
        "retrieved genre and mood knowledge above. Address: "
        "(1) how well results match the request, "
        "(2) any catalog limitations or guardrail warnings, "
        "(3) confidence level (low/medium/high), "
        "(4) one concrete improvement suggestion."
    )

    messages = [{"role": "user", "content": initial_prompt}]  # conversation history, grows with each turn
    recommendations = None         # populated after the tool executes
    tool_call_summary = "(no tool call)"   # overwritten if Claude issues a tool call

    log.info("Step 3: Agentic call — Claude invoking get_song_recommendations tool")

    # --- Turn 1: send the profile to Claude; expect it to respond with a tool_use block ---
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,     # enough for Claude to produce the tool call JSON
        tools=[tool_def],    # makes get_song_recommendations available to Claude
        messages=messages,
    )

    if response.stop_reason == "tool_use":
        # Claude issued a tool call — extract the tool_use block from the response
        tool_block = next(b for b in response.content if b.type == "tool_use")
        inp = tool_block.input   # dict of parameters Claude chose (genre, mood, energy, etc.)

        # Build a human-readable summary of the tool call for logging and terminal output
        tool_call_summary = (
            f"get_song_recommendations("
            f"genre={inp.get('genre')!r}, "
            f"mood={inp.get('mood')!r}, "
            f"target_energy={inp.get('target_energy')}, "
            f"likes_acoustic={inp.get('likes_acoustic')}, "
            f"k={inp.get('k', k)})"
        )
        log.info("Tool called: %s", tool_call_summary)

        # Execute the tool deterministically using the rule-based engine.
        # Claude's parameters take priority; fall back to the parsed profile if any field is missing.
        tool_profile = {
            "genre":          inp.get("genre",          profile["genre"]),
            "mood":           inp.get("mood",           profile["mood"]),
            "target_energy":  inp.get("target_energy",  profile["target_energy"]),
            "likes_acoustic": inp.get("likes_acoustic", profile["likes_acoustic"]),
        }
        recommendations = recommend_songs(tool_profile, songs, k=inp.get("k", k))

        # Format the tool result as a plain-text string to send back in Turn 2
        rec_text = "\n".join(
            f"#{i + 1} {s['title']} by {s['artist']} "
            f"(genre={s['genre']}, mood={s['mood']}, energy={s['energy']}, score={sc:.2f}/6.5)"
            for i, (s, sc, _) in enumerate(recommendations)
        )
        log.info(
            "Tool returned %d results; top=%r score=%.2f",
            len(recommendations),
            recommendations[0][0]["title"] if recommendations else "N/A",
            recommendations[0][1] if recommendations else 0.0,
        )

        # --- Turn 2: append assistant response + tool result, then ask for the streaming critique ---
        messages.append({"role": "assistant", "content": response.content})  # Claude's Turn 1 response
        messages.append({
            "role": "user",
            # tool_result block is required by the API when replying to a tool_use block
            "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": rec_text}],
        })

        parts: List[str] = []   # collect streaming text chunks
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=512,    # 4–6 sentence critique fits in 512 tokens
            messages=messages, # full 2-turn conversation: system+user → tool_call → tool_result
        ) as stream:
            for text in stream.text_stream:   # each chunk is a partial token
                parts.append(text)
        critique = "".join(parts)   # reassemble full critique from streaming chunks
        log.info("Step 4: Critique complete (%d chars)", len(critique))

    else:
        # Fallback path: Claude responded in plain text without calling the tool.
        # This should not happen in normal operation but is handled gracefully.
        log.warning("Claude did not issue a tool call; falling back to direct recommendation")
        critique = next(
            (b.text for b in response.content if hasattr(b, "text")),
            "No critique generated.",
        )
        recommendations = recommend_songs(profile, songs, k=k)  # run engine directly as fallback

    return recommendations, critique, tool_call_summary   # caller unpacks all three values


# ---------------------------------------------------------------------------
# Demo mode — pre-scripted profiles and critiques (no API key required)
# ---------------------------------------------------------------------------

_DEMO_PROFILES: Dict[str, Dict] = {
    "study": {
        "genre": "lofi", "mood": "focused", "target_energy": 0.35,
        "likes_acoustic": True, "confidence": 0.91, "notes": None,
    },
    "workout": {
        "genre": "rock", "mood": "intense", "target_energy": 0.88,
        "likes_acoustic": False, "confidence": 0.93, "notes": None,
    },
    "reggae": {
        "genre": "r&b", "mood": "intense", "target_energy": 0.85,
        "likes_acoustic": False, "confidence": 0.62,
        "notes": "Reggae not in catalog. Mapped to closest catalog genre: r&b.",
    },
}

_DEMO_CRITIQUES: Dict[str, str] = {
    "study": (
        "Focus Flow is an excellent match. It falls within lofi's typical energy range\n"
        "of 0.2-0.5 and satisfies the focused mood's guidance exactly: low energy,\n"
        "mid-range valence, and acoustic character keep it in the background without\n"
        "interrupting concentration. The 98% score reflects alignment across all five\n"
        "scoring signals - a rare result. The remaining lofi songs (Library Rain,\n"
        "Midnight Coding) continue the pattern but reveal a catalog limitation: only\n"
        "three lofi tracks are available, so positions four and five draw from adjacent\n"
        "quiet genres (jazz, ambient). Confidence: high. Expanding the lofi catalog\n"
        "would be the single most impactful improvement."
    ),
    "workout": (
        "Storm Runner leads convincingly - rock at 0.91 energy is a near-perfect fit\n"
        "for an intense workout profile. The genre and mood bonuses both fire, and\n"
        "energy closeness is high. Positions two through four show the catalog's\n"
        "high-energy bias: metal and EDM tracks cluster here because their energy\n"
        "scores are competitive even without a genre match. Confidence: high. The\n"
        "results are functionally correct for this profile."
    ),
    "reggae": (
        "No reggae songs exist in the catalog - the system correctly flagged\n"
        "GENRE_NOT_IN_CATALOG and mapped to r&b as the closest match via the\n"
        "knowledge base. The top r&b result earns the genre bonus but the mood\n"
        "mismatch is visible in the score gap. Positions two through five fill with\n"
        "high-energy songs from other genres. Confidence: medium. This is a catalog\n"
        "coverage problem - the system behaved correctly given its data, but cannot\n"
        "serve reggae requests well until reggae tracks are added."
    ),
}


def _pick_demo_key(user_text: str) -> str:
    t = user_text.lower()
    if any(w in t for w in ("reggae", "caribbean", "island")):
        return "reggae"
    if any(w in t for w in ("workout", "run", "gym", "exercise", "morning run")):
        return "workout"
    return "study"


def run_demo_session(user_text: str, songs_path: str = _SONGS_PATH, k: int = 5) -> None:
    """
    Demo pipeline: identical output format to run_agentic_session() but uses
    pre-scripted Claude responses so no API key is needed. The rule engine,
    guardrails, and session logging all run for real.
    """
    session_id = new_session_id()
    log = get_logger(session_id)
    log.info("=== Demo session started | request: %s", user_text)

    try:
        songs = load_songs(songs_path)
    except FileNotFoundError:
        print(f"\n[ERROR] Cannot find songs catalog: {songs_path}")
        return

    kb = load_knowledge_base()
    kb_moods = load_mood_knowledge_base()
    catalog_genres = {s["genre"] for s in songs}

    demo_key = _pick_demo_key(user_text)
    dp = _DEMO_PROFILES[demo_key]
    critique = _DEMO_CRITIQUES[demo_key]

    profile = {
        "genre": dp["genre"],
        "mood": dp["mood"],
        "target_energy": dp["target_energy"],
        "likes_acoustic": dp["likes_acoustic"],
    }

    width = 65
    sep = "=" * width
    thin = "-" * width

    print(f"\n{sep}")
    print(f"  AI MUSIC RECOMMENDER - Agentic Session [{session_id}]")
    print(sep)
    print(f'\n  Request: "{user_text}"\n')

    print("  [1/4] Parsing request with Claude (RAG + few-shot grounded)...")
    print("  Parsed profile:")
    print(f"    Genre      : {profile['genre']}   Mood : {profile['mood']}")
    print(f"    Energy     : {profile['target_energy']:.2f}   Acoustic : {profile['likes_acoustic']}")
    print(f"    Confidence : {dp['confidence']:.0%}")
    if dp["notes"]:
        print(f"    Notes      : {dp['notes']}")
    log.info(
        "Demo profile: genre=%s mood=%s energy=%.2f acoustic=%s confidence=%.0f%%",
        profile["genre"], profile["mood"], profile["target_energy"],
        profile["likes_acoustic"], dp["confidence"] * 100,
    )

    print(f"\n  [2/4] Running guardrails...")
    issues = run_guardrails(profile, songs)
    for issue in issues:
        if issue.severity == Severity.ERROR:
            log.error("Guardrail ERROR — %s: %s", issue.code, issue.message)
        elif issue.severity == Severity.WARNING:
            log.warning("Guardrail WARNING — %s: %s", issue.code, issue.message)
        else:
            log.info("Guardrail INFO — %s: %s", issue.code, issue.message)

    genre_gap = next((i for i in issues if i.code == "GENRE_NOT_IN_CATALOG"), None)
    if genre_gap:
        closest = find_closest_catalog_genre(profile["genre"], catalog_genres, kb)
        if closest:
            log.info("RAG fallback: '%s' -> '%s'", profile["genre"], closest)
            print(f"  [RAG] Genre '{profile['genre']}' not in catalog.")
            print(f"        Knowledge base suggests closest match: '{closest}'")

    if issues:
        print(format_issues(issues))
    else:
        print("  No issues detected.")

    print(f"\n  [3/4] Agentic: Claude calling recommendation tool...")
    _ = format_genre_context(profile["genre"], kb)    # RAG retrieval ② (logged, not displayed)
    _ = format_mood_context(profile["mood"], kb_moods)  # RAG retrieval ③
    log.debug("RAG: retrieved genre context for '%s'", profile["genre"])
    log.debug("RAG: retrieved mood context for '%s'", profile["mood"])

    recommendations = recommend_songs(profile, songs, k=k)
    tool_call_summary = (
        f"get_song_recommendations(genre='{profile['genre']}', "
        f"mood='{profile['mood']}', "
        f"target_energy={profile['target_energy']}, "
        f"likes_acoustic={profile['likes_acoustic']}, k={k})"
    )
    log.info("Demo tool call: %s", tool_call_summary)
    print(f"  >> Tool call: {tool_call_summary}")
    print(f"  << Tool returned {len(recommendations)} songs")

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

    print(f"\n  [4/4] Claude self-critique (dual-RAG: genre + mood)...")
    print(f"  {thin}")
    for line in critique.strip().splitlines():
        print(f"  {line}")
    log.info("=== Demo session completed")
    print(f"  {thin}")
    print(f"\n  Log written to: logs/sessions.log  [session={session_id}]")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Full agentic session — orchestrates all 4 steps
# ---------------------------------------------------------------------------

def run_agentic_session(
    user_text: str,
    songs_path: str = _SONGS_PATH,  # defaults to data/songs.csv relative to this file
    k: int = 5,                     # number of recommendations to return
) -> None:
    """
    Full pipeline: free-text input -> RAG-grounded NL parse -> guardrails
    -> rule engine -> RAG-grounded Claude critique -> printed output.

    Logs every step to logs/sessions.log.
    """
    session_id = new_session_id()   # 8-char hex UUID — unique per run, used in log lines
    log = get_logger(session_id)    # session-scoped logger writing to logs/sessions.log

    # Fail fast if no API key — avoids cryptic SDK errors later
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY environment variable is not set")
        print("\n[ERROR] ANTHROPIC_API_KEY is not set.")
        print("  Set it with:  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    log.info("=== Session started | request: %s", user_text)

    # Load song catalog — fail gracefully if the CSV is missing
    try:
        songs = load_songs(songs_path)
    except FileNotFoundError:
        log.error("Songs catalog not found at: %s", songs_path)
        print(f"\n[ERROR] Cannot find songs catalog: {songs_path}")
        return

    log.info("Catalog loaded: %d songs", len(songs))

    # Load both knowledge bases upfront — used at multiple points in the pipeline
    kb = load_knowledge_base()            # knowledge/genres.json → genre profiles
    kb_moods = load_mood_knowledge_base() # knowledge/moods.json  → mood profiles

    # Derive the set of genres that actually exist in the catalog — used for guardrails
    # and for the RAG catalog overview (only lists genres present in songs.csv)
    catalog_genres = {s["genre"] for s in songs}

    # RAG retrieval ①: build the genre table injected into the NL parsing prompt
    catalog_overview = format_catalog_overview(catalog_genres, kb)
    log.debug("RAG: catalog overview retrieved (%d genres)", len(catalog_genres))

    # Instantiate the Anthropic client once; reused for all API calls in this session
    client = anthropic.Anthropic(api_key=api_key)

    # Visual formatting constants
    width = 65
    sep = "=" * width
    thin = "-" * width

    print(f"\n{sep}")
    print(f"  AI MUSIC RECOMMENDER - Agentic Session [{session_id}]")
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

    # Convert Pydantic object to a plain dict used by the rest of the pipeline
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

    # Log a warning when confidence is below 50% — signal that the profile may be unreliable
    if extraction.extraction_confidence < 0.5:
        log.warning(
            "Low extraction confidence (%.0f%%) — profile may not reflect user intent",
            extraction.extraction_confidence * 100,
        )

    # ------------------------------------------------------------------
    # Step 2: Guardrails (pre-flight checks before any scoring)
    # ------------------------------------------------------------------
    print(f"\n  [2/4] Running guardrails...")
    issues = run_guardrails(profile, songs)   # returns list of GuardrailIssue objects

    # Log each issue at the appropriate severity level
    for issue in issues:
        if issue.severity == Severity.ERROR:
            log.error("Guardrail ERROR — %s: %s", issue.code, issue.message)
        elif issue.severity == Severity.WARNING:
            log.warning("Guardrail WARNING — %s: %s", issue.code, issue.message)
        else:
            log.info("Guardrail INFO — %s: %s", issue.code, issue.message)

    # RAG fallback: if the genre isn't in the catalog, look up the closest catalog genre
    # using the similar_genres list in knowledge/genres.json
    genre_gap = next((i for i in issues if i.code == "GENRE_NOT_IN_CATALOG"), None)
    if genre_gap:
        closest = find_closest_catalog_genre(profile["genre"], catalog_genres, kb)
        if closest:
            log.info("RAG fallback: '%s' not in catalog, closest match is '%s'", profile["genre"], closest)
            print(f"  [RAG] Genre '{profile['genre']}' not in catalog.")
            print(f"        Knowledge base suggests closest match: '{closest}'")

    # Print guardrail results — format_issues() renders severity icons and suggestions
    if issues:
        print(format_issues(issues))
    else:
        print("  No issues detected.")

    # ------------------------------------------------------------------
    # Steps 3+4: Agentic tool call + dual-RAG critique
    # ------------------------------------------------------------------
    print(f"\n  [3/4] Agentic: Claude calling recommendation tool...")

    # RAG retrieval ②+③: retrieve both genre and mood context before the agentic call.
    # Both are injected into the critique prompt so Claude can ground its analysis
    # in factual energy/valence ranges rather than general knowledge.
    genre_context = format_genre_context(profile["genre"], kb)
    mood_context = format_mood_context(profile["mood"], kb_moods)
    log.debug("RAG: retrieved genre context for '%s'", profile["genre"])
    log.debug("RAG: retrieved mood context for '%s'", profile["mood"])

    try:
        recommendations, critique, tool_call_summary = run_agentic_recommendation_and_critique(
            user_text, profile, issues, genre_context, mood_context, songs, client, log, k=k
        )
    except anthropic.APIError:
        print("  [ERROR] Agentic call failed. Check logs/sessions.log.")
        log.info("=== Session ended with error (Steps 3+4)")
        return

    # Print the observable intermediate step — shows exactly what parameters Claude chose
    print(f"  >> Tool call: {tool_call_summary}")
    print(f"  << Tool returned {len(recommendations)} songs")

    # Print ranked recommendations with visual score bar
    print(f"\n  Top {k} Recommendations:")
    for rank, (song, score, explanation) in enumerate(recommendations, 1):
        pct = score / 6.5                                   # normalize to 0–1 for the bar
        bar = "#" * int(pct * 10) + "-" * (10 - int(pct * 10))  # 10-char ASCII bar
        print(f"\n    #{rank}  {song['title']} by {song['artist']}")
        print(f"         Score  : {score:.2f}/6.5  [{bar}] {pct:.0%} match")
        print(f"         Genre  : {song['genre']} | Mood: {song['mood']} | Energy: {song['energy']}")
        print("         Why    :")
        for reason in explanation.split("; "):   # explanation is semicolon-delimited reasons
            print(f"           - {reason}")

    # ------------------------------------------------------------------
    # Step 4: Print streaming critique
    # ------------------------------------------------------------------
    print(f"\n  [4/4] Claude self-critique (dual-RAG: genre + mood)...")
    print(f"  {thin}")
    for line in critique.strip().splitlines():
        print(f"  {line}")   # indent each line to align with session output
    log.info("=== Session completed successfully")
    print(f"  {thin}")

    print(f"\n  Log written to: logs/sessions.log  [session={session_id}]")
    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo = "--demo" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--demo"]
    runner = run_demo_session if demo else run_agentic_session

    if args:
        user_text = " ".join(args)
        runner(user_text)
    else:
        mode = "Demo" if demo else "Agentic"
        print(f"\nAI Music Recommender - {mode} Mode")
        print("Type your music request in plain English. Press Ctrl+C to quit.\n")
        try:
            while True:
                user_text = input("Your request: ").strip()
                if not user_text:
                    continue
                runner(user_text)
                print()
        except KeyboardInterrupt:
            print("\nGoodbye.")


if __name__ == "__main__":
    main()
