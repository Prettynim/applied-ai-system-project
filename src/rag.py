"""
RAG (Retrieval-Augmented Generation) module for the Music Recommender.

Retrieves genre knowledge from a structured knowledge base BEFORE Claude generates
any response. This retrieved context is injected into Claude's prompts so that
Claude's outputs are grounded in factual genre data rather than general knowledge.

Two integration points:

1. NL Parsing (Step 1):
   The full catalog genre overview is retrieved and included in the system prompt.
   This helps Claude pick the closest valid catalog genre even for vague or
   off-catalog requests (e.g., "something like reggae" maps to folk or r&b).

2. Self-Critique (Step 4):
   The genre profile for the TOP recommended song is retrieved and included.
   Claude uses this to compare actual recommendations against expected genre
   characteristics and identify mismatches (e.g., a folk song scoring high for
   an energy=0.9 request despite folk typical energy being 0.2-0.55).

Knowledge base: knowledge/genres.json
"""

import json
import os
from typing import Optional, Dict, Any, Set

_KB_PATH = os.path.join(os.path.dirname(__file__), "..", "knowledge", "genres.json")


def load_knowledge_base(path: str = _KB_PATH) -> Dict[str, Any]:
    """Load the genre knowledge base from disk. Returns empty dict on failure."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def retrieve_genre_context(genre: str, kb: Optional[Dict] = None) -> Optional[Dict]:
    """
    Look up a single genre's metadata from the knowledge base.
    Returns the genre dict or None if the genre is not in the knowledge base.
    """
    if kb is None:
        kb = load_knowledge_base()
    return kb.get(genre.lower())


def find_closest_catalog_genre(
    requested_genre: str,
    catalog_genres: Set[str],
    kb: Optional[Dict] = None,
) -> Optional[str]:
    """
    Given a genre NOT in the catalog, find the closest catalog genre
    by walking the 'similar_genres' list in the knowledge base.

    Returns the first similar genre that exists in the catalog, or None.
    """
    if kb is None:
        kb = load_knowledge_base()

    genre_data = kb.get(requested_genre.lower(), {})
    for candidate in genre_data.get("similar_genres", []):
        if candidate in catalog_genres:
            return candidate
    return None


def format_genre_context(genre: str, kb: Optional[Dict] = None) -> str:
    """
    Format a single genre's characteristics as a readable string for
    inclusion in a Claude prompt. Used during the self-critique step.
    """
    if kb is None:
        kb = load_knowledge_base()

    data = kb.get(genre.lower())
    if not data:
        return f"No knowledge base entry found for genre '{genre}'."

    lo, hi = data.get("typical_energy_range", ["?", "?"])
    moods = ", ".join(data.get("typical_moods", []))
    similar = ", ".join(data.get("similar_genres", []))
    acoustic = "yes" if data.get("acoustic_common") else "no"

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
    Used during NL parsing so Claude knows what's actually available before
    it maps a vague request to a specific genre.
    """
    if kb is None:
        kb = load_knowledge_base()

    lines = ["RETRIEVED KNOWLEDGE — Available catalog genres:"]
    for genre in sorted(catalog_genres):
        data = kb.get(genre.lower(), {})
        if data:
            lo, hi = data.get("typical_energy_range", ["?", "?"])
            moods = data.get("typical_moods", [])[:2]
            mood_str = ", ".join(moods) if moods else "varies"
            lines.append(f"  {genre:<12} energy {lo}–{hi}   moods: {mood_str}")
        else:
            lines.append(f"  {genre:<12} (no knowledge base entry)")
    return "\n".join(lines)
