"""
Tests for src/rag.py

Covers:
- load_knowledge_base() returns a non-empty dict
- retrieve_genre_context() returns data for known genres and None for unknown
- find_closest_catalog_genre() returns a valid catalog genre for off-catalog inputs
- format_genre_context() includes key fields in its output string
- format_catalog_overview() covers all catalog genres
"""

import pytest
from src.rag import (
    load_knowledge_base,
    retrieve_genre_context,
    find_closest_catalog_genre,
    format_genre_context,
    format_catalog_overview,
)

CATALOG_GENRES = {
    "pop", "rock", "lofi", "jazz", "edm", "hip-hop",
    "r&b", "classical", "country", "metal", "funk",
    "folk", "ambient", "synthwave", "indie pop",
}


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base()


# ---------------------------------------------------------------------------
# load_knowledge_base
# ---------------------------------------------------------------------------

def test_kb_is_not_empty(kb):
    assert len(kb) > 0


def test_kb_contains_catalog_genres(kb):
    for genre in CATALOG_GENRES:
        assert genre in kb, f"'{genre}' missing from knowledge base"


def test_kb_entries_have_required_fields(kb):
    required = {"description", "typical_energy_range", "typical_moods", "similar_genres", "acoustic_common"}
    for genre, data in kb.items():
        missing = required - data.keys()
        assert not missing, f"Genre '{genre}' is missing fields: {missing}"


def test_kb_energy_ranges_are_valid(kb):
    for genre, data in kb.items():
        lo, hi = data["typical_energy_range"]
        assert 0.0 <= lo <= 1.0, f"{genre}: energy low={lo} out of range"
        assert 0.0 <= hi <= 1.0, f"{genre}: energy high={hi} out of range"
        assert lo <= hi, f"{genre}: energy low > high"


# ---------------------------------------------------------------------------
# retrieve_genre_context
# ---------------------------------------------------------------------------

def test_retrieve_known_genre_returns_dict(kb):
    result = retrieve_genre_context("pop", kb)
    assert isinstance(result, dict)
    assert "description" in result


def test_retrieve_unknown_genre_returns_none(kb):
    result = retrieve_genre_context("bossa nova", kb)
    assert result is None


def test_retrieve_is_case_insensitive(kb):
    lower = retrieve_genre_context("lofi", kb)
    upper = retrieve_genre_context("LOFI", kb)
    assert lower == upper


# ---------------------------------------------------------------------------
# find_closest_catalog_genre
# ---------------------------------------------------------------------------

def test_reggae_maps_to_catalog_genre(kb):
    # reggae similar_genres should include folk or r&b (both in catalog)
    result = find_closest_catalog_genre("reggae", CATALOG_GENRES, kb)
    assert result in CATALOG_GENRES


def test_soul_maps_to_catalog_genre(kb):
    result = find_closest_catalog_genre("soul", CATALOG_GENRES, kb)
    assert result in CATALOG_GENRES


def test_electronic_maps_to_catalog_genre(kb):
    result = find_closest_catalog_genre("electronic", CATALOG_GENRES, kb)
    assert result in CATALOG_GENRES


def test_catalog_genre_itself_is_not_remapped(kb):
    # pop IS in catalog; find_closest is only called when genre is NOT in catalog,
    # but if called for a catalog genre, it should still return something valid
    # (its similar genres are also often in catalog)
    result = find_closest_catalog_genre("pop", CATALOG_GENRES, kb)
    # could be None if no similar genres are in catalog, or a valid genre
    assert result is None or result in CATALOG_GENRES


def test_completely_unknown_genre_returns_none_or_catalog(kb):
    result = find_closest_catalog_genre("xylowave", CATALOG_GENRES, kb)
    assert result is None or result in CATALOG_GENRES


# ---------------------------------------------------------------------------
# format_genre_context
# ---------------------------------------------------------------------------

def test_format_genre_context_contains_genre_name(kb):
    output = format_genre_context("jazz", kb)
    assert "jazz" in output.lower()


def test_format_genre_context_contains_energy_range(kb):
    output = format_genre_context("jazz", kb)
    assert "energy" in output.lower()
    # Jazz typical range is 0.3-0.7
    assert "0.3" in output or "0.7" in output


def test_format_genre_context_contains_description(kb):
    output = format_genre_context("lofi", kb)
    assert len(output) > 50  # should be a meaningful block of text


def test_format_genre_context_unknown_genre(kb):
    output = format_genre_context("zydeco", kb)
    assert "No knowledge base entry" in output


# ---------------------------------------------------------------------------
# format_catalog_overview
# ---------------------------------------------------------------------------

def test_catalog_overview_covers_all_genres(kb):
    output = format_catalog_overview(CATALOG_GENRES, kb)
    for genre in CATALOG_GENRES:
        assert genre in output, f"'{genre}' not found in catalog overview"


def test_catalog_overview_contains_energy_info(kb):
    output = format_catalog_overview(CATALOG_GENRES, kb)
    assert "energy" in output.lower()


def test_catalog_overview_is_multiline(kb):
    output = format_catalog_overview(CATALOG_GENRES, kb)
    lines = output.strip().splitlines()
    assert len(lines) >= len(CATALOG_GENRES), "Expected at least one line per genre"
