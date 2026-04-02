# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

This simulation builds a content-based music recommender that scores each song in a small catalog against a user's stated taste profile. It prioritizes genre and mood as the strongest signals, then uses the closeness of a song's energy level to the user's target energy, with optional bonuses for acoustic preference and valence. The top-k highest-scoring songs are returned as recommendations, along with a plain-language explanation of why each song was chosen.

---

## How The System Works

Real-world recommenders like Spotify or YouTube use two main strategies: **collaborative filtering**, which finds patterns across millions of users ("people like you also liked..."), and **content-based filtering**, which matches a song's measurable attributes to a user's known preferences. At scale these are combined into hybrid systems, but they all share the same core idea: turn both the user and every song into numbers, then find the closest match.

This simulation uses a **pure content-based approach**. Every song and every user is represented as a set of features. The recommender computes a weighted score for each song, then ranks them and returns the top results. Genre and mood mismatches are treated as hard penalties (they carry the most weight), while numeric features like energy are rewarded for closeness to the user's target rather than for simply being high or low. This mirrors how real systems reward relevance over raw popularity.

### Song features

Each `Song` object stores:

- `genre` — broad stylistic category (pop, lofi, rock, ambient, jazz, synthwave, indie pop)
- `mood` — emotional tone (happy, chill, intense, relaxed, moody, focused)
- `energy` — 0.0 to 1.0, how loud and active the track feels
- `tempo_bpm` — beats per minute
- `valence` — 0.0 to 1.0, musical positiveness (high = upbeat, low = dark)
- `danceability` — 0.0 to 1.0, rhythmic suitability for dancing
- `acousticness` — 0.0 to 1.0, acoustic vs. electronic/produced sound

### UserProfile features

Each `UserProfile` stores:

- `favorite_genre` — the genre the user most wants to hear
- `favorite_mood` — the emotional tone they are looking for right now
- `target_energy` — their preferred energy level on a 0.0 to 1.0 scale
- `likes_acoustic` — boolean flag, true if they prefer acoustic over electronic sound

### Scoring and ranking

1. **Score each song** using a weighted formula:
   - Genre match: +2.0 points (binary)
   - Mood match: +1.5 points (binary)
   - Energy closeness: `(1 - |target_energy - song.energy|) x 1.5`
   - Acoustic bonus: +1.0 if `likes_acoustic` is true and `song.acousticness > 0.6`
   - Valence closeness: `(1 - |0.65 - song.valence|) x 0.5`
2. **Rank all songs** by score, highest first
3. **Return the top k songs** along with a plain-language explanation for each

### Data flow diagram

```mermaid
flowchart TD
    A([User Profile\ngenre · mood · target_energy · likes_acoustic]) --> C
    B[(data/songs.csv\n18 songs)] --> C

    C[load_songs: read CSV into memory] --> D

    D{For each song\nin catalog}

    D --> E[Score the song]
    E --> E1[+2.0 if genre matches]
    E --> E2[+1.5 if mood matches]
    E --> E3[+1.5 x energy closeness\n1 - abs target - song.energy]
    E --> E4[+1.0 if likes_acoustic\nand acousticness > 0.6]
    E --> E5[+0.5 x valence closeness\n1 - abs 0.65 - song.valence]

    E1 & E2 & E3 & E4 & E5 --> F[Sum = total score\nmax 6.5]

    F --> G{More songs\nremaining?}
    G -- Yes --> D
    G -- No --> H

    H[Sort all scored songs\nhighest score first]
    H --> I[Take top k results]
    I --> J([Output: Ranked Recommendations\ntitle · score · explanation])
```

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Sample Terminal Output

Running with the default **Pop / Happy** profile (`python main.py` from `src/`):

```
Catalog loaded: 18 songs

============================================================
  Profile : Pop / Happy
  Genre   : pop   Mood: happy
  Energy  : 0.8   Acoustic: False
============================================================

  #1  Sunrise City by Neon Echo
       Score : 5.38 / 6.5
       Genre : pop  |  Mood: happy  |  Energy: 0.82
       Why   :
         - genre match - pop (+2.0)
         - mood match - happy (+1.5)
         - energy 0.82 vs target 0.8 -> closeness 0.98 (+1.47)
         - valence 0.84 -> closeness 0.81 (+0.41)

  #2  Gym Hero by Max Pulse
       Score : 3.74 / 6.5
       Genre : pop  |  Mood: intense  |  Energy: 0.93
       Why   :
         - genre match - pop (+2.0)
         - energy 0.93 vs target 0.8 -> closeness 0.87 (+1.3)
         - valence 0.77 -> closeness 0.88 (+0.44)

  #3  Rooftop Lights by Indigo Parade
       Score : 3.36 / 6.5
       Genre : indie pop  |  Mood: happy  |  Energy: 0.76
       Why   :
         - mood match - happy (+1.5)
         - energy 0.76 vs target 0.8 -> closeness 0.96 (+1.44)
         - valence 0.81 -> closeness 0.84 (+0.42)

  #4  Crown Up by Verse Capital
       Score : 1.94 / 6.5
       Genre : hip-hop  |  Mood: confident  |  Energy: 0.78
       Why   :
         - energy 0.78 vs target 0.8 -> closeness 0.98 (+1.47)
         - valence 0.72 -> closeness 0.93 (+0.47)

  #5  Night Drive Loop by Neon Echo
       Score : 1.84 / 6.5
       Genre : synthwave  |  Mood: moody  |  Energy: 0.75
       Why   :
         - energy 0.75 vs target 0.8 -> closeness 0.95 (+1.42)
         - valence 0.49 -> closeness 0.84 (+0.42)

------------------------------------------------------------
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

### Phase 4 Stress Test — 9 Profiles (5 normal + 4 adversarial)

All profiles were run against the 18-song catalog. Key observations:

**Normal profiles — behaved as expected**

| Profile | Top result | Score | Notes |
|---|---|---|---|
| Pop / Happy | Sunrise City | 5.38 | Genre + mood double match, near-perfect energy |
| Study Session | Focus Flow | 6.44 | Near-max score; only lofi song with mood=focused |
| Workout | Storm Runner | 5.39 | Only rock/intense song; clear winner |
| Sunday Morning | Coffee Shop Stories | 6.44 | Only jazz/relaxed song; near-max |
| Dance Floor | Signal Drop | 5.42 | Only EDM/euphoric song; perfect energy match |

**Adversarial profiles — revealed system weaknesses**

**Edge 1 — Unknown genre (`reggae`):**
Genre bonus is permanently locked out (max possible score = 4.5). The system falls back to mood + energy. Result: two "happy" songs topped the list even though neither is remotely reggae. Shows the system has no way to say "no catalog match found" — it always recommends something.

**Edge 2 — High-energy acoustic (`folk/intense/0.9`):**
The only folk song (Porch Light) ranked #1 at 3.99 — but it has energy=0.32, far from the target of 0.9. Genre weight (2.0) + acoustic bonus (1.0) outweighed the energy penalty. Songs #2 and #3 (Gym Hero, Storm Runner) scored nearly as high by matching mood=intense and energy, but with no genre match. The system was "tricked" into recommending a quiet folk song as #1 for an intense listener.

**Edge 3 — Dead-center energy (`synthwave/moody/0.5`):**
Night Drive Loop still won cleanly (5.04) via genre + mood match. But positions #2–#5 were a confused cluster within 0.14 points of each other — country, R&B, lofi, lofi — with scores around 1.84–1.96. No song clusters near 0.5 energy, so the long tail is essentially random.

**Edge 4 — Impossible mood (`lofi/romantic`):**
No lofi song in the catalog has mood=romantic, so the mood bonus (+1.5) can never be collected. The top 3 results were all lofi songs winning purely on genre + energy + acoustic. Golden Hour (R&B/romantic) ranked #4 by collecting the mood bonus despite genre mismatch — scoring 3.20 vs the lofi songs at ~4.9. This shows the system heavily favors genre over mood, which can frustrate a user whose primary need is a specific emotional tone.

```
Catalog loaded: 18 songs

============================================================
  Profile : Edge: Unknown Genre
  Genre   : reggae   Mood: happy
  Energy  : 0.65   Acoustic: False
============================================================

  #1  Rooftop Lights by Indigo Parade
       Score : 3.25 / 6.5  (max possible without genre match: 4.5)
       Genre : indie pop  |  Mood: happy  |  Energy: 0.76
       Why   :
         - mood match - happy (+1.5)
         - energy 0.76 vs target 0.65 -> closeness 0.89 (+1.33)
         - valence 0.81 -> closeness 0.84 (+0.42)

============================================================
  Profile : Edge: High-Energy Acoustic
  Genre   : folk   Mood: intense
  Energy  : 0.9   Acoustic: True
============================================================

  #1  Porch Light by Elm & Ash   <- quiet folk song ranked #1 for "intense" listener
       Score : 3.99 / 6.5
       Genre : folk  |  Mood: melancholic  |  Energy: 0.32
       Why   :
         - genre match - folk (+2.0)
         - energy 0.32 vs target 0.9 -> closeness 0.42 (+0.63)   <- heavy penalty
         - acoustic match - acousticness 0.88 (+1.0)

============================================================
  Profile : Edge: Impossible Mood
  Genre   : lofi   Mood: romantic
  Energy  : 0.4   Acoustic: True
============================================================

  #1  Focus Flow by LoRoom   <- correct genre, wrong mood, still wins
       Score : 4.97 / 6.5
  #4  Golden Hour by Sienna Vale   <- has the right mood, wrong genre, ranks 4th
       Score : 3.20 / 6.5
       Why   : mood match - romantic (+1.5) but no genre bonus
```

### Phase 4 Step 3 — Weight Shift Experiment

**Change applied:** genre weight halved (2.0 → 1.0), energy weight doubled (1.5 → 3.0). New max score: 7.0.

**Before vs. after — key ranking changes:**

| Profile | Original #1→#2 | Experimental #1→#2 | What changed |
|---|---|---|---|
| Pop / Happy | Sunrise City → **Gym Hero** | Sunrise City → **Rooftop Lights** | Rooftop Lights (indie pop/happy) jumped to #2; Gym Hero dropped to #3 |
| High-Energy Acoustic | **Porch Light** → Gym Hero | **Storm Runner** → Gym Hero | Quiet folk song no longer wins; mood+energy beats genre |
| Impossible Mood | Focus Flow → … → **Golden Hour #4** | Focus Flow → … → **Golden Hour #4** | Same structure; gap narrowed (5.47 vs 4.48 instead of 4.97 vs 3.20) |

**Was the change more accurate or just different?**

Mixed result — better in some places, worse in others:

- **Better — Pop/Happy:** Rooftop Lights (indie pop/happy) now ranks #2 instead of Gym Hero (pop/intense). Musically, a happy indie pop song is a better recommendation for a "happy pop" user than an intense gym anthem. The fix worked.
- **Better — High-Energy Acoustic:** Porch Light (a quiet, melancholic folk track) dropped from #1 to #3. Storm Runner and Gym Hero — both actually intense — now rank ahead of it. Energy's larger weight correctly penalized the energy mismatch.
- **Worse — long tail noise:** With reduced genre weight, the #2–#5 slots in profiles like Dance Floor and Dead-Center Energy filled with random high-energy songs (Iron Veil, Sunrise City) that have no mood or genre match. Energy became so dominant it surfaced songs that were simply "close in BPM" regardless of feel.
- **Unchanged weakness:** The fundamental problem (genre is an exact string match) was not fixed by reweighting. "Indie pop" still scores 0 for a "pop" user. The experiment revealed that the right fix is **genre similarity grouping**, not just lowering the genre weight.

**Conclusion:** The original weights (genre 2.0, energy 1.5) were restored because the experiment created new problems while solving old ones. The most impactful improvement would be adding genre similarity groups (e.g., `pop` and `indie pop` share partial credit) rather than adjusting numeric weights.

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

[**Model Card**](model_card.md) | [**Profile Comparisons**](reflection.md)

### Biggest learning moment

The biggest learning moment came during the adversarial testing phase, specifically the High-Energy Acoustic profile. I expected a user who wanted intense, high-energy folk music to receive energetic songs. Instead, the system ranked Porch Light — a quiet, melancholic song with energy 0.32 — as the top result. The math was not broken. Genre match (2.0) plus acoustic bonus (1.0) simply outweighed the energy penalty, fair and square by the formula's rules. That moment made something concrete that had been abstract: **a system can be mathematically correct and still wrong in a way that matters to a real person.** Every weight I assigned was a guess about what users value most, and when those guesses were wrong, the system failed confidently rather than cautiously.

### How recommenders turn data into predictions

Before building this, a music recommendation felt almost magical — the app just *knew* what I wanted. After building one, it looks like a loop that runs 18 times, adds up some numbers, and sorts a list. What creates the feeling of intelligence is not complexity but specificity: when the scores happen to align with what a person would have chosen themselves, the math feels like taste. When they do not align — like getting a gym track when you asked for happy pop — the illusion breaks. Real platforms like Spotify have millions of songs and behavioral signals (skips, replays, saves) that tighten the guesses over time. This simulation has 18 songs and no feedback loop, which makes the gaps obvious. That gap is instructive.

### Where bias showed up

Bias did not appear as a dramatic mistake. It appeared as a series of small, unconsidered choices. Setting the valence anchor at 0.65 felt like a neutral default, but it quietly penalizes dark-music listeners every time the formula runs. Using exact string matching for genre felt like the obvious implementation, but it makes "indie pop" invisible to a "pop" user. Putting 50% of songs above energy 0.7 felt like building a diverse catalog, but it structurally disadvantages mid-energy listeners who can never get a perfect energy score. None of these were intentional. They were the result of thinking about the average case and not asking who falls outside it. In a system used by real people at scale, those quiet penalties would accumulate into real unfairness — certain listeners consistently getting worse recommendations than others, for no reason they could see or contest.

### Using AI tools — where they helped and where I checked

AI tools were most useful for two things: generating the initial data (the 8 new songs added to `songs.csv`) and explaining tradeoffs between design options, like why `.sort()` is safer than `sorted()` for a list you own, or why energy closeness should reward proximity rather than magnitude. The output was trustworthy for these kinds of structured reasoning tasks.

Where I needed to double-check: when AI suggested a scoring formula, I verified the math by computing a perfect-match case manually before using it. When AI explained that "genre weight might be too strong," I ran the weight-shift experiment rather than just changing the number — and discovered the suggestion was partially right but would create new problems it had not anticipated. The pattern was consistent: AI explanations were reliable as starting points and reliable for debugging, but experimental changes needed to be verified by running the actual code and reading the actual output.

### What would come next

The single highest-value extension would be replacing exact genre string matching with a genre similarity map — a small lookup table where "indie pop" and "pop" share partial credit, "rock" and "metal" share partial credit, and so on. This would fix the most frustrating failure mode without adding any new complexity to the scoring architecture. Second priority would be adding a `target_valence` field to `UserProfile` so listeners who prefer dark or melancholic music are no longer silently pushed toward brighter songs. Both changes would take about 15 minutes of code and would make the system feel meaningfully smarter for a much wider range of users.


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

