"""
RAG (Retrieval-Augmented Generation) module for the Music Recommender.

Retrieves genre and mood knowledge from structured knowledge bases BEFORE Claude
generates any response. This retrieved context is injected into Claude's prompts
so that outputs are grounded in factual data rather than general knowledge.

Knowledge bases:
  knowledge/genres.json  — 21 genre profiles (energy ranges, moods, acousticness)
  knowledge/moods.json   — 14 mood profiles (energy/valence ranges, compatible genres)

Three integration points:

1. NL Parsing (Step 1):
   The full catalog genre overview is retrieved and included in the system prompt.
   This helps Claude pick the closest valid catalog genre even for vague or
   off-catalog requests (e.g., "something like reggae" maps to folk or r&b).

2. Agentic Critique — Genre (Step 4):
   The genre profile for the requested genre is retrieved and passed alongside
   the tool result. Claude uses this to identify energy/mood mismatches against
   expected genre characteristics.

3. Agentic Critique — Mood (Step 4):
   The mood profile for the requested mood is retrieved from moods.json and
   passed alongside the genre context. Dual-RAG grounding lets Claude compare
   recommendations against both genre AND mood expected ranges simultaneously.
"""

import json    # standard library — used to parse the JSON knowledge base files
import os      # used to build absolute paths relative to this file's location
from typing import Optional, Dict, Any, Set   # type hints for all public functions

# Path to the genre knowledge base, resolved relative to this file — not the working directory.
# Using __file__ means the path works regardless of where the user runs the script from.
_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "genres.json")


def load_knowledge_base(path: str = _KB_PATH) -> Dict[str, Any]:
    """
    Load the genre knowledge base from disk.
    Returns an empty dict on failure so callers degrade gracefully
    instead of crashing when the file is missing or malformed.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)   # parse JSON into a Python dict keyed by genre name
    except (FileNotFoundError, json.JSONDecodeError):
        return {}   # silent failure — callers check for missing keys individually


def retrieve_genre_context(genre: str, kb: Optional[Dict] = None) -> Optional[Dict]:
    """
    Look up a single genre's metadata from the knowledge base.
    Returns the genre dict or None if the genre is not in the knowledge base.

    `kb` can be passed in to avoid re-loading the file on every call.
    If omitted, the knowledge base is loaded fresh from disk.
    """
    if kb is None:
        kb = load_knowledge_base()    # lazy-load if no kb passed — convenient for one-off calls
    return kb.get(genre.lower())      # normalize to lowercase so "Lofi" and "lofi" both match


def find_closest_catalog_genre(
    requested_genre: str,
    catalog_genres: Set[str],    # set of genres that actually exist in songs.csv
    kb: Optional[Dict] = None,
) -> Optional[str]:
    """
    RAG fallback: given a genre NOT in the catalog, find the closest catalog genre
    by walking the 'similar_genres' list in the knowledge base.

    Returns the first similar genre that exists in the catalog, or None.
    Used in Step 2 (guardrails) when GENRE_NOT_IN_CATALOG fires — lets the system
    suggest an alternative before the user sees a 0-bonus result.
    """
    if kb is None:
        kb = load_knowledge_base()

    genre_data = kb.get(requested_genre.lower(), {})   # look up the off-catalog genre's entry
    for candidate in genre_data.get("similar_genres", []):   # walk similar_genres list in order
        if candidate in catalog_genres:   # stop at the first match that's actually in the catalog
            return candidate
    return None   # no similar genre exists in the catalog


def format_genre_context(genre: str, kb: Optional[Dict] = None) -> str:
    """
    Format a single genre's characteristics as a readable string for
    inclusion in a Claude prompt. Used during the agentic critique step (Step 4).

    The formatted output anchors Claude's critique to factual data:
    e.g. "lofi typical energy: 0.2–0.5" prevents Claude from making up ranges.
    """
    if kb is None:
        kb = load_knowledge_base()

    data = kb.get(genre.lower())    # retrieve the genre entry; None if not found
    if not data:
        return f"No knowledge base entry found for genre '{genre}'."   # safe fallback message

    lo, hi = data.get("typical_energy_range", ["?", "?"])    # energy bounds for this genre
    moods = ", ".join(data.get("typical_moods", []))          # comma-separated mood list
    similar = ", ".join(data.get("similar_genres", []))       # comma-separated similar genres
    acoustic = "yes" if data.get("acoustic_common") else "no" # boolean → human-readable

    # RETRIEVED CONTEXT header signals to Claude (and readers) that this is injected data,
    # not something Claude generated — important for attribution and prompt clarity
    return (
        f"RETRIEVED CONTEXT — Genre: {genre}\n"
        f"  Description     : {data.get('description', 'N/A')}\n"
        f"  Typical energy  : {lo} – {hi}\n"
        f"  Typical moods   : {moods}\n"
        f"  Similar genres  : {similar}\n"
        f"  Acoustic common : {acoustic}"
    )


def format_catalog_overview(catalog_genres: Set[str], kb: Optional[Dict] = None) -> str:
    """
    Build a brief table of all catalog genres with their energy ranges and moods.

    Used during NL parsing (Step 1) so Claude knows what's actually available before
    it maps a vague user request to a specific genre. Without this table, Claude
    might map "reggae" to "reggae" even though reggae isn't in the catalog.

    Only genres present in catalog_genres are included — so the table stays in
    sync with songs.csv automatically, even if genres are added or removed.
    """
    if kb is None:
        kb = load_knowledge_base()

    lines = ["RETRIEVED KNOWLEDGE — Available catalog genres:"]
    for genre in sorted(catalog_genres):    # alphabetical order for consistent prompt layout
        data = kb.get(genre.lower(), {})    # look up this catalog genre in the knowledge base
        if data:
            lo, hi = data.get("typical_energy_range", ["?", "?"])
            moods = data.get("typical_moods", [])[:2]   # show only top 2 moods to keep the table compact
            mood_str = ", ".join(moods) if moods else "varies"
            lines.append(f"  {genre:<12} energy {lo}–{hi}   moods: {mood_str}")
        else:
            # Genre exists in the catalog but has no KB entry — warn Claude so it doesn't fabricate data
            lines.append(f"  {genre:<12} (no knowledge base entry)")
    return "\n".join(lines)   # single multi-line string ready to embed in a prompt


# ---------------------------------------------------------------------------
# Mood knowledge base (second RAG source — stretch feature)
# ---------------------------------------------------------------------------

# Path to the mood knowledge base, resolved the same way as the genre path
_MOODS_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "moods.json")


def load_mood_knowledge_base(path: str = _MOODS_KB_PATH) -> Dict[str, Any]:
    """
    Load the mood knowledge base from disk. Returns empty dict on failure.
    Mirrors load_knowledge_base() exactly — same error-handling pattern.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)   # parse JSON into dict keyed by mood name
    except (FileNotFoundError, json.JSONDecodeError):
        return {}   # degrade gracefully if moods.json is missing


def format_mood_context(mood: str, kb_moods: Optional[Dict] = None) -> str:
    """
    Format a single mood's characteristics as a readable string for
    inclusion in a Claude prompt. Used during the agentic critique step
    alongside format_genre_context() to provide dual-RAG grounding.

    The critique_guidance field is the key addition over the genre context:
    it tells Claude exactly what to check in the recommendations (e.g.,
    "verify energy > 0.75 and valence < 0.4 for angry mood").
    """
    if kb_moods is None:
        kb_moods = load_mood_knowledge_base()   # lazy-load if not provided

    data = kb_moods.get(mood.lower())   # normalize to lowercase for consistent lookup
    if not data:
        return f"No mood knowledge base entry found for mood '{mood}'."   # safe fallback

    e_lo, e_hi = data.get("typical_energy_range", ["?", "?"])    # expected energy bounds
    v_lo, v_hi = data.get("typical_valence_range", ["?", "?"])   # expected valence bounds
    genres = ", ".join(data.get("compatible_genres", []))         # genres that pair well with this mood

    # RETRIEVED CONTEXT header signals this is injected knowledge, not Claude's general knowledge
    return (
        f"RETRIEVED CONTEXT — Mood: {mood}\n"
        f"  Description      : {data.get('description', 'N/A')}\n"
        f"  Typical energy   : {e_lo} – {e_hi}\n"
        f"  Typical valence  : {v_lo} – {v_hi}\n"
        f"  Compatible genres: {genres}\n"
        f"  Critique guidance: {data.get('critique_guidance', 'N/A')}"   # tells Claude what mismatches to look for
    )
