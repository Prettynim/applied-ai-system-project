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

    # --- Edge Case / Adversarial Profiles ---

    # Edge 1: Genre not in catalog — system can never award genre bonus
    unknown_genre = {
        "genre": "reggae",
        "mood": "happy",
        "target_energy": 0.65,
        "likes_acoustic": False,
    }

    # Edge 2: Conflicting preferences — high energy but loves acoustic instruments
    # (high-energy acoustic songs are rare; tests whether energy or acousticness wins)
    high_energy_acoustic = {
        "genre": "folk",
        "mood": "intense",
        "target_energy": 0.90,
        "likes_acoustic": True,
    }

    # Edge 3: Dead-center energy (0.5) — no song clusters near the middle,
    # so energy closeness never pays off fully for anyone
    middle_energy = {
        "genre": "synthwave",
        "mood": "moody",
        "target_energy": 0.50,
        "likes_acoustic": False,
    }

    # Edge 4: Genre exists, but matching mood does not exist in catalog
    # (lofi/romantic — no song has that combo; mood bonus is permanently locked out)
    impossible_mood = {
        "genre": "lofi",
        "mood": "romantic",
        "target_energy": 0.40,
        "likes_acoustic": True,
    }

    # Run all profiles in sequence
    all_profiles = [
        ("Pop / Happy",            pop_happy),
        ("Study Session",          study_session),
        ("Workout",                workout),
        ("Sunday Morning",         sunday_morning),
        ("Dance Floor",            dance_floor),
        ("Edge: Unknown Genre",    unknown_genre),
        ("Edge: High-Energy Acoustic", high_energy_acoustic),
        ("Edge: Dead-Center Energy",   middle_energy),
        ("Edge: Impossible Mood",      impossible_mood),
    ]

    for name, profile in all_profiles:
        results = recommend_songs(profile, songs, k=5)
        print_recommendations(name, profile, results)


if __name__ == "__main__":
    main()
