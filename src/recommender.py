import csv                                          # standard library — reads songs.csv
from typing import List, Dict, Tuple, Optional     # type hints for all public functions
from dataclasses import dataclass                  # lightweight data containers for OOP interface

@dataclass
class Song:
    """
    Typed representation of a single catalog song.
    Required by tests/test_recommender.py — the OOP test suite uses this dataclass
    rather than raw dicts so fields can be accessed by name (song.genre vs song["genre"]).
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float       # 0.0 (very quiet/calm) to 1.0 (very loud/intense)
    tempo_bpm: float    # beats per minute — loaded but not used in scoring (future feature)
    valence: float      # 0.0 (sad/dark) to 1.0 (happy/bright)
    danceability: float # loaded but not used in scoring (future feature)
    acousticness: float # 0.0 (fully electronic) to 1.0 (fully acoustic)

@dataclass
class UserProfile:
    """
    Typed representation of a user's music preferences.
    Required by tests/test_recommender.py for the OOP Recommender class interface.
    The agent pipeline uses plain dicts instead — both representations are supported.
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float   # target energy level 0.0–1.0
    likes_acoustic: bool   # True = user prefers acoustic-sounding music

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against a user preference dict.
    Returns (total_score, reasons) where reasons is a list of
    human-readable strings explaining each point contribution.

    Scoring formula (max 6.5 pts total):
      genre match    : +2.0  — strongest signal; exact string match only
      mood match     : +1.5  — context signal; exact string match only
      energy close   : +1.5  — continuous; rewards proximity to target
      acoustic match : +1.0  — flat bonus when user prefers acoustic AND song qualifies
      valence close  : +0.5  — fine-tuning signal; anchored to 0.65 (slightly positive)

    user_prefs keys: genre, mood, target_energy, likes_acoustic
    song keys: genre, mood, energy, valence, acousticness (plus others)
    """
    score = 0.0    # accumulator — starts at zero, each check adds points
    reasons = []   # list of human-readable strings, one per scoring signal

    # Genre match: +2.0 pts (highest weight)
    # Exact string comparison only — "indie pop" and "pop" earn zero shared credit.
    # This is a known limitation: the weight is high because genre is the single
    # strongest signal for user satisfaction in content-based filtering.
    if song["genre"] == user_prefs["genre"]:
        score += 2.0
        reasons.append(f"genre match - {song['genre']} (+2.0)")

    # Mood match: +1.5 pts (second highest)
    # Also exact string comparison — "happy" and "euphoric" earn zero shared credit.
    if song["mood"] == user_prefs["mood"]:
        score += 1.5
        reasons.append(f"mood match - {song['mood']} (+1.5)")

    # Energy closeness: up to +1.5 pts (continuous signal)
    # Formula: (1.0 - |target - song_energy|) * 1.5
    # A perfect match (distance=0) earns 1.5; a maximum mismatch (distance=1) earns 0.
    # This linear interpolation rewards proximity without binary thresholds.
    energy_closeness = 1.0 - abs(user_prefs["target_energy"] - song["energy"])
    energy_points = round(energy_closeness * 1.5, 2)   # round to 2dp to avoid floating-point noise
    score += energy_points
    reasons.append(
        f"energy {song['energy']} vs target {user_prefs['target_energy']} "
        f"-> closeness {energy_closeness:.2f} (+{energy_points})"
    )

    # Acoustic bonus: +1.0 pt (flat, conditional)
    # Only awarded when BOTH conditions hold: user prefers acoustic AND song is acoustic (> 0.6).
    # The 0.6 threshold selects songs that are clearly acoustic-dominant.
    # Note: there is NO equivalent bonus for users who prefer electronic music — a known asymmetry.
    if user_prefs["likes_acoustic"] and song["acousticness"] > 0.6:
        score += 1.0
        reasons.append(f"acoustic match - acousticness {song['acousticness']} (+1.0)")

    # Valence closeness: up to +0.5 pts (fine-tuning signal)
    # The anchor is fixed at 0.65 (slightly positive) — not the user's requested valence.
    # This silently rewards positive-sounding songs regardless of what the user asked for.
    # Known limitation: users who prefer dark/melancholic music are penalized by this anchor.
    valence_closeness = 1.0 - abs(0.65 - song["valence"])   # distance from the fixed anchor 0.65
    valence_points = round(valence_closeness * 0.5, 2)       # scale to 0–0.5 range
    score += valence_points
    reasons.append(f"valence {song['valence']} -> closeness {valence_closeness:.2f} (+{valence_points})")

    return round(score, 2), reasons   # round total to 2dp; return alongside explanation list


class Recommender:
    """
    OOP wrapper around the scoring logic.
    Required by tests/test_recommender.py — provides a class-based interface
    that mirrors how real recommendation systems are typically structured.
    The agent pipeline uses the module-level recommend_songs() function instead.
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs   # catalog is injected at construction time

    def _score(self, user: UserProfile, song: Song) -> float:
        """
        Adapter: converts UserProfile + Song dataclasses to plain dicts
        and delegates to the module-level score_song() function.
        Keeps the dataclass interface compatible with the dict-based engine.
        """
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dict = {
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "valence": song.valence,
            "acousticness": song.acousticness,
        }
        total, _ = score_song(user_prefs, song_dict)   # discard reasons — only need the score
        return total

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """
        Returns the top-k songs ranked by score for the given user profile.
        sorted() with reverse=True produces descending order (highest score first).
        """
        scored = sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)
        return scored[:k]   # slice to top-k only

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """
        Returns a semicolon-separated string of scoring reasons for one song.
        Used by tests to verify that the explanation format is correct.
        """
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "target_energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dict = {
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "valence": song.valence,
            "acousticness": song.acousticness,
        }
        _, reasons = score_song(user_prefs, song_dict)   # only need reasons here
        return "; ".join(reasons)   # join into a single string for display

def load_songs(csv_path: str) -> List[Dict]:
    """
    Reads songs.csv and returns a list of dicts with numeric fields cast to the
    correct Python types (int / float). csv.DictReader returns all values as strings
    by default — without casting, energy comparisons like `0.8 > song["energy"]`
    would compare a float against a string and always return False.
    """
    int_fields = {"id", "tempo_bpm"}           # fields that should be integers
    float_fields = {"energy", "valence", "danceability", "acousticness"}  # fields that should be floats

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)   # reads header row automatically; yields one dict per data row
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])      # cast string → int
            for field in float_fields:
                row[field] = float(row[field])    # cast string → float
            songs.append(row)   # add the fully-typed dict to the list
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Scores every song against user_prefs and returns the top-k as (song, score, explanation) tuples.

    This is the primary entry point used by the agent pipeline and the evaluator.
    The function is deterministic: same inputs always produce the same ranked output.
    The Recommender class above provides an OOP alternative for test compatibility.
    """
    scored = []
    for song in songs:
        total, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)    # flatten reasons list to a semicolon-delimited string
        scored.append((song, total, explanation))

    # Sort descending by score — highest-scoring songs appear first
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]   # return only the top-k results
