# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Goal / Task

VibeFinder 1.0 tries to predict which songs a listener will enjoy right now, based on what they tell it about their taste. Given a user's favorite genre, current mood, preferred energy level, and whether they like acoustic sound, the system ranks every song in an 18-song catalog from best match to worst and returns the top 5. It does not learn over time — it makes a one-shot prediction from the user's stated profile.

---

## 3. Data Used

The catalog contains **18 songs** described by 10 features each: id, title, artist, genre, mood, energy (0–1 scale), tempo in BPM, valence (0–1, emotional brightness), danceability (0–1), and acousticness (0–1). The 18 songs span 15 genres — lofi, pop, rock, jazz, ambient, synthwave, indie pop, hip-hop, R&B, classical, country, metal, funk, folk, and EDM — and 14 moods. The original starter file had 10 songs; 8 were added to improve genre diversity.

**Limits of the data:**
- 18 songs is too small for a real recommender. Many genres and moods have exactly one song, so there is no variety within a category.
- The catalog reflects a Western, English-language view of popular music. Non-Western genres such as Afrobeats, K-pop, reggaeton, or bossa nova are entirely absent.
- 50% of songs have energy above 0.7. Mid-energy listeners (target around 0.5) are underserved because no songs cluster in that range.
- Only 4 of 18 songs have valence below 0.5, meaning dark and melancholic music is underrepresented.

---

## 4. Algorithm Summary

The system works in two steps: **scoring** and **ranking**.

**Scoring** assigns a number to each song based on how well it matches the user. Five things are checked:

1. Does the song's genre match the user's favorite genre? If yes, +2 points. This is the biggest single bonus because genre is the sharpest dividing line in music taste.
2. Does the song's mood match the user's target mood? If yes, +1.5 points. Mood captures why someone is listening — to study, to work out, to relax — which matters as much as genre.
3. How close is the song's energy to the user's target energy? A song with exactly the right energy scores up to +1.5 points; a song at the opposite extreme scores close to 0. Closeness is rewarded, not just "high energy" or "low energy."
4. Does the user prefer acoustic sound, and is the song mostly acoustic? If both are true, +1 point.
5. How emotionally bright is the song (valence)? Songs near a neutral brightness (0.65 out of 1.0) earn up to +0.5 points. This is the smallest bonus — a fine-tuning signal.

The maximum possible score is **6.5 points**.

**Ranking** is straightforward: sort all 18 scored songs from highest to lowest, then return the top 5. Each recommendation also prints a plain-language explanation showing exactly which of the five checks contributed points and how many.

---

## 5. Observed Behavior and Biases

**What works well:** When a user's preferred genre has at least one song that also matches their mood, that song wins clearly. The lofi/focused listener always gets Focus Flow first (6.44/6.5). The jazz/relaxed listener always gets Coffee Shop Stories first (6.44/6.5). The scoring is fully explainable — every recommendation includes a reason string so the user can see exactly why a song appeared.

**Bias 1 — Genre string matching is too strict.** "Indie pop" and "pop" are treated as completely different by the system because they are different text strings. A pop fan whose favorite tracks happen to be labeled "indie pop" earns zero genre bonus. In testing, Rooftop Lights (indie pop/happy) ranked #3 for a Pop/Happy listener even though it was more emotionally appropriate than the #2 result.

**Bias 2 — High-energy songs dominate the long tail.** Half the catalog has energy above 0.7. Whenever a profile has no strong genre or mood match, the long-tail positions fill with high-energy songs regardless of genre or mood, because energy closeness can earn up to 1.5 points. "Gym Hero" appeared in the top five of four different profiles for this reason.

**Bias 3 — Valence is anchored to 0.65 without asking the user.** The formula silently assumes all users prefer slightly positive music. A listener who loves dark, heavy, or sad music is quietly penalized every time a low-valence song is scored, because the formula pushes scores toward the middle of the brightness range.

**Bias 4 — Filter bubble.** There is no diversity rule. A lofi user always gets lofi songs in their top three, with no mechanism to surface adjacent genres they might enjoy. The system optimizes for match, not for discovery.

**Bias 5 — Acoustic users get a scoring advantage.** The acoustic bonus (+1.0) is available only to users who prefer acoustic sound. Users who prefer electronic music cannot earn this point, making their long-tail results structurally weaker.

---

## 6. Evaluation Process

Nine user profiles were tested against the 18-song catalog: five standard profiles (Pop/Happy, Study Session, Workout, Sunday Morning, Dance Floor) and four adversarial profiles designed to expose failures (Unknown Genre, High-Energy Acoustic, Dead-Center Energy, Impossible Mood).

**Standard profiles** all produced intuitive #1 results — in each case the single song matching both genre and mood ranked first with a gap of at least 1.5 points over the runner-up.

**Adversarial profiles** revealed the system's main weaknesses:
- The *Unknown Genre* profile (reggae, not in catalog) could never earn genre points, reducing the max possible score to 4.5. The system still returned results without warning the user that nothing truly matched.
- The *High-Energy Acoustic* profile (folk, intense, energy=0.9) ranked Porch Light — a quiet, melancholic folk song with energy=0.32 — as #1, because the genre bonus (2.0) plus acoustic bonus (1.0) outweighed the severe energy penalty. The math was correct; the result felt wrong.
- The *Dead-Center Energy* profile (target=0.5) produced a confident #1 winner (Night Drive Loop, which matched genre and mood) but positions #2–#5 were a near-tied cluster of country, R&B, and lofi songs — effectively random, because no catalog song has energy near 0.5.
- The *Impossible Mood* profile (lofi, romantic) showed that genre dominates mood: the top three results were all lofi songs despite none of them being romantic, while the only romantic song (Golden Hour, R&B) ranked #4.

One controlled experiment was run: genre weight halved (2.0→1.0), energy weight doubled (1.5→3.0). This fixed the High-Energy Acoustic problem and improved Pop/Happy rankings, but created new noise in high-energy profiles. The original weights were restored. The experiment showed that reweighting is a blunt instrument — the underlying problem is exact string matching for genre, not the numeric values of weights.

Full profile comparisons in plain language are in [reflection.md](reflection.md).

---

## 7. Intended Use and Non-Intended Use

**Intended use:**
- Learning how content-based recommender systems work by building one from scratch
- Exploring how weighted scoring translates user preferences into ranked results
- Understanding what "bias" and "filter bubble" mean in a concrete, small-scale system
- Classroom demonstration and discussion of AI transparency and explainability

**Not intended for:**
- Real music discovery by actual users — the catalog is too small and the genre matching too rigid to provide useful variety
- Any deployment outside this classroom project
- Making decisions about what music artists, labels, or genres to promote — the catalog and scoring weights embed choices that would be unfair at scale
- Representing how Spotify, YouTube, or any real platform actually works — real systems use learned embeddings, collaborative filtering, and billions of behavioral data points that this simulation does not have

---

## 8. Ideas for Improvement

**1. Genre similarity groupings instead of exact string matching.**
Replace `song.genre == user.genre` with a lookup table that awards partial credit for related genres (e.g., pop and indie pop share 0.5 points, rock and metal share 0.5 points). This would fix the most common real-world labeling problem and reduce the filter bubble effect for users whose taste spans related genres.

**2. User-controlled valence target.**
Add a `target_valence` field to `UserProfile` (similar to `target_energy`) and replace the fixed 0.65 anchor in the formula. This would let listeners who prefer dark, melancholic music receive recommendations that match their emotional preference instead of being silently pushed toward brighter songs.

**3. Diversity rule in the ranking step.**
After scoring, prevent more than two songs from the same genre from appearing in the top five. This would force the system to surface adjacent genres even when a user's exact match dominates the catalog, reducing the filter bubble and giving users exposure to music they might not have known to ask for.

---

## 9. Personal Reflection

Building this recommender made the invisible math behind streaming platforms suddenly visible. What felt like a "smart" system is actually a small set of weighted if-statements running in a loop — and that simplicity is both the system's strength (it is fully explainable) and its greatest weakness (it cannot learn, adapt, or handle anything outside its fixed rules).

The most unexpected discovery was how dramatically a single weight change could flip rankings, and yet still leave the core problem — exact string matching for genre — completely untouched. This suggests that the choice of *how features are represented* matters far more than tuning numeric weights. A genre similarity map would have fixed more problems than any amount of weight adjustment.

The valence anchor was a bias I did not notice I had introduced until a systematic audit. It was baked in as a shortcut and it silently disadvantages listeners who prefer dark music. This is a good example of how bias in AI systems often comes not from bad intentions but from small, unconsidered design choices that seem neutral at the time. Real music recommenders almost certainly use learned embeddings rather than exact string matches precisely because no two people mean the same thing when they say they like "pop."
