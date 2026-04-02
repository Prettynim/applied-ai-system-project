import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against a user preference dict.
    Returns (total_score, reasons) where reasons is a list of
    human-readable strings explaining each point contribution.

    user_prefs keys: genre, mood, target_energy, likes_acoustic
    song keys: genre, mood, energy, valence, acousticness (plus others)
    """
    score = 0.0
    reasons = []

    # Genre match: +2.0 (strongest categorical signal)
    if song["genre"] == user_prefs["genre"]:
        score += 2.0
        reasons.append(f"genre match - {song['genre']} (+2.0)")

    # Mood match: +1.5 (context signal)
    if song["mood"] == user_prefs["mood"]:
        score += 1.5
        reasons.append(f"mood match - {song['mood']} (+1.5)")

    # Energy closeness: up to +1.5 (rewards proximity, not magnitude)
    energy_closeness = 1.0 - abs(user_prefs["target_energy"] - song["energy"])
    energy_points = round(energy_closeness * 1.5, 2)
    score += energy_points
    reasons.append(
        f"energy {song['energy']} vs target {user_prefs['target_energy']} "
        f"-> closeness {energy_closeness:.2f} (+{energy_points})"
    )

    # Acoustic bonus: +1.0 (flat bonus when user prefers acoustic sound)
    if user_prefs["likes_acoustic"] and song["acousticness"] > 0.6:
        score += 1.0
        reasons.append(f"acoustic match - acousticness {song['acousticness']} (+1.0)")

    # Valence closeness: up to +0.5 (fine-tunes emotional brightness)
    valence_closeness = 1.0 - abs(0.65 - song["valence"])
    valence_points = round(valence_closeness * 0.5, 2)
    score += valence_points
    reasons.append(f"valence {song['valence']} -> closeness {valence_closeness:.2f} (+{valence_points})")

    return round(score, 2), reasons


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def _score(self, user: UserProfile, song: Song) -> float:
        """Converts UserProfile + Song dataclasses to dicts and calls score_song."""
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
        total, _ = score_song(user_prefs, song_dict)
        return total

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Returns the top-k songs ranked by score for the given user profile."""
        scored = sorted(self.songs, key=lambda s: self._score(user, s), reverse=True)
        return scored[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Returns a semicolon-separated string of scoring reasons for one song."""
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
        _, reasons = score_song(user_prefs, song_dict)
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Reads songs.csv and returns a list of dicts with numeric fields cast to int/float."""
    int_fields = {"id", "tempo_bpm"}
    float_fields = {"energy", "valence", "danceability", "acousticness"}

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in int_fields:
                row[field] = int(row[field])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Scores every song against user_prefs and returns the top-k as (song, score, explanation) tuples."""
    scored = []
    for song in songs:
        total, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored.append((song, total, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
