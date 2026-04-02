"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def print_recommendations(profile_name: str, profile: dict, results: list) -> None:
    """Prints a clean, labeled recommendation block to the terminal."""
    width = 60
    print("\n" + "=" * width)
    print(f"  Profile : {profile_name}")
    print(f"  Genre   : {profile['genre']}   Mood: {profile['mood']}")
    print(f"  Energy  : {profile['target_energy']}   Acoustic: {profile['likes_acoustic']}")
    print("=" * width)

    for rank, (song, score, explanation) in enumerate(results, start=1):
        print(f"\n  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Score : {score:.2f} / 6.5")
        print(f"       Genre : {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
        print("       Why   :")
        for reason in explanation.split("; "):
            print(f"         - {reason}")

    print("\n" + "-" * width)


def main() -> None:
    songs = load_songs("../data/songs.csv")
    print(f"Catalog loaded: {len(songs)} songs")

    # --- User Profiles ---
    # Swap the value assigned to `active_profile` to test different personas.

    pop_happy = {
        "genre": "pop",
        "mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }

    study_session = {
        "genre": "lofi",
        "mood": "focused",
        "target_energy": 0.38,
        "likes_acoustic": True,
    }

    workout = {
        "genre": "rock",
        "mood": "intense",
        "target_energy": 0.92,
        "likes_acoustic": False,
    }

    sunday_morning = {
        "genre": "jazz",
        "mood": "relaxed",
        "target_energy": 0.35,
        "likes_acoustic": True,
    }

    dance_floor = {
        "genre": "edm",
        "mood": "euphoric",
        "target_energy": 0.95,
        "likes_acoustic": False,
    }

    # Active profile — change this line to switch personas
    active_profile = pop_happy
    active_name = "Pop / Happy"

    results = recommend_songs(active_profile, songs, k=5)
    print_recommendations(active_name, active_profile, results)


if __name__ == "__main__":
    main()
