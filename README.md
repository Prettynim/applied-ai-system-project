# AI Music Recommender

An AI-powered music recommendation system that accepts free-text requests, retrieves supporting knowledge, scores a catalog of songs, and then critiques its own results — all in a single agentic pipeline built on the Claude API.

**GitHub:** https://github.com/Prettynim/applied-ai-system-project  
**Video walkthrough:** *(Loom link — add after recording)*

---

## Origin: Module 1–3 Project

This project extends **VibeFinder**, a content-based music recommender built during Modules 1–3. The original system represented songs and user taste profiles as structured data, then ranked every song in an 18-song catalog using a weighted scoring formula (genre, mood, energy, acoustic preference, and valence). It demonstrated how real-world platforms like Spotify convert user preferences into ranked lists, and was evaluated against five standard profiles and four adversarial edge cases.

In this final project, VibeFinder is extended with a Claude API layer, a RAG knowledge base, guardrails, session logging, and a reliability test suite — transforming a standalone scoring script into a full applied AI system.

---

## What This System Does

A user types a request in plain English: *"I want something calm for studying"* or *"give me high-energy workout music."* The system:

1. **Parses** the request with Claude (few-shot grounded) to extract a structured preference profile
2. **Retrieves** genre knowledge from `knowledge/genres.json` before the parse (RAG retrieval ①)
3. **Checks** the profile with five guardrail rules before any scoring begins
4. **Retrieves** both genre and mood context from two knowledge bases (RAG retrieval ② and ③)
5. **Scores** songs via Claude tool call — Claude calls `get_song_recommendations`, the rule engine executes, results return as a tool result (observable agentic step)
6. **Critiques** the results using Claude, grounded in the dual-retrieved context

Every step is logged to `logs/sessions.log` with a unique session ID. A separate evaluator runs 9 test cases offline to measure reliability without consuming API tokens.

---

## System Architecture

![System architecture diagram](assets/Architecture.png)

The system has three distinct pathways:

**Agentic pipeline** (`src/agent.py`) — the main user-facing flow. A free-text request enters at the top. Two RAG retrievals from `knowledge/genres.json` inject factual context before each Claude call: the first injects the full catalog overview to ground genre selection; the second injects the top-result's genre profile to ground the critique. Between those Claude calls, a deterministic rule engine (`src/recommender.py`) scores songs from `data/songs.csv`. The result is a ranked list plus a streaming critique.

**Guardrails** (`src/guardrails.py`) — five pre-scoring safety checks that run between NL parsing and the rule engine. They flag catalog gaps (genre not in catalog, mood not in catalog), input contradictions (high energy + acoustic preference), data quality issues (thin genre coverage), and range violations (energy outside 0–1). If a genre gap is detected, a RAG fallback looks up the closest catalog genre.

**Reliability testing** (`src/evaluator.py`) — a separate, offline test suite. Nine test cases cover normal profiles, catalog-gap profiles, adversarial inputs, and a determinism check. It produces a percentage reliability score and letter grade. A developer reviews these results before relying on the system.

---

## Required AI Features

| Feature | Where it lives | How it's integrated |
|---|---|---|
| **RAG** | `src/rag.py` + `knowledge/genres.json` + `knowledge/moods.json` | Genre data is retrieved before Claude parses the request. Both genre and mood data are retrieved before Claude critiques results. All Claude outputs are anchored to retrieved facts, not general knowledge. |
| **Agentic workflow** | `src/agent.py` | Pipeline: Claude parses → guardrails check → **Claude calls `get_song_recommendations` tool** → rule engine executes → tool result returned → Claude critiques. The recommendation step is an observable tool call, not a hidden function call. |
| **Reliability testing** | `src/evaluator.py` | 9 structured test cases with programmatic pass/fail checks. Covers correctness, guardrail behavior, and determinism. Produces a scored report. |
| **Guardrails + logging** | `src/guardrails.py`, `src/logger.py` | Pre-flight checks before every recommendation run. All steps, API calls, and guardrail events logged to `logs/sessions.log` with session IDs. |

---

## Stretch Features

| Feature | What was built | Points |
|---|---|---|
| **RAG Enhancement** | Added `knowledge/moods.json` as a second data source (14 mood profiles with energy ranges, valence ranges, compatible genres, critique guidance). Both genre context AND mood context are retrieved and injected into the critique prompt — Claude can now identify when a recommended song's valence or energy contradicts the requested mood. | +2 |
| **Agentic Workflow Enhancement** | Replaced the direct `recommend_songs()` call with a Claude tool-use step. Claude receives the profile and decides to call `get_song_recommendations` — the tool call parameters, execution, and result are all observable. This makes the recommendation decision an explicit intermediate step in a 2-turn agentic loop. | +2 |
| **Fine-Tuning / Specialization** | Added `_FEW_SHOT_EXAMPLES` to the NL parsing system prompt: 4 calibration examples showing correct confidence levels (0.90 for clear requests, 0.35 for vague ones, 0.62 for off-catalog genres) and the expected genre-remapping note format. Claude's extractions now follow consistent calibration patterns instead of drifting on ambiguous inputs. | +2 |
| **Test Harness** | `src/evaluator.py` runs 9 predefined test cases covering normal profiles, catalog gaps, adversarial inputs, and determinism. Produces a pass/fail table, confidence scores per case, and an overall reliability score with letter grade. Runs without an API key. | +2 |

### Stretch Feature Demonstrations

**RAG Enhancement — dual-source impact**

Without `moods.json` (single-source RAG), the critique can only reference genre data:
> *"Focus Flow aligns with lofi's typical energy range of 0.2–0.5."*

With `moods.json` (dual-source RAG), the critique also references mood data:
> *"Focus Flow aligns with lofi's typical energy range of 0.2–0.5 and satisfies the focused mood's guidance: typical energy 0.2–0.5, valence 0.3–0.65 — keeping the music in the background without interrupting concentration."*

The second critique identifies a valence-range match that the first cannot, because mood characteristics were not retrieved.

**Fine-Tuning / Specialization — before and after few-shot**

Without few-shot examples, a vague request like *"something nice"* produces unstable confidence (may be 0.6–0.8 with no notes, overconfident for what is actually an underspecified input).

With four few-shot calibration examples in the system prompt, the same request consistently produces:
```
Confidence : 35%
Notes      : Very vague request. Defaulted to ambient/relaxed as most
             neutral background profile. Genre and mood may not reflect intent.
```

The 35% confidence matches the calibration example for vague requests directly. Off-catalog genre requests consistently produce ~0.62 confidence with a remapping note, matching the reggae example in the few-shot set. This pattern holds across sessions because the examples anchor the model's calibration rather than letting it vary freely.

---

## Setup

### 1. Get an Anthropic API key

The agentic pipeline requires a Claude API key. Get one at [console.anthropic.com](https://console.anthropic.com).

Set it as an environment variable before running the agent:

```bash
# Mac / Linux
export ANTHROPIC_API_KEY=sk-ant-...

# Windows Command Prompt
set ANTHROPIC_API_KEY=sk-ant-...

# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

> The rule-based runner (`src.main`) and evaluator (`src.evaluator`) do not require an API key.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the System

### AI-powered agent (primary entry point)

```bash
# Interactive mode — type requests one at a time, Ctrl+C to quit
python -m src.agent

# Single-request mode
python -m src.agent "I want something calm and focused for studying"
```

### Reliability evaluator (no API key needed)

```bash
python -m src.evaluator
```

### Rule-based runner (no API key needed)

```bash
python -m src.main
```

### Tests

```bash
pytest
```

---

## Sample Interactions

The following examples show representative output from `python -m src.agent`. Claude's critique text varies between sessions; scores and rankings are fully deterministic.

---

### Example 1 — Clear, specific request

**Input:**
```
Your request: I want something calm and focused for studying, maybe acoustic
```

**Output:**
```
=================================================================
  AI MUSIC RECOMMENDER — Agentic Session [3a8f21c0]
=================================================================

  Request: "I want something calm and focused for studying, maybe acoustic"

  [1/4] Parsing request with Claude (RAG + few-shot grounded)...
  Parsed profile:
    Genre      : lofi   Mood : focused
    Energy     : 0.35   Acoustic : True
    Confidence : 91%

  [2/4] Running guardrails...
  No issues detected.

  [3/4] Agentic: Claude calling recommendation tool...
  → Tool call: get_song_recommendations(genre='lofi', mood='focused',
               target_energy=0.35, likes_acoustic=True, k=5)
  ← Tool returned 5 songs

  Top 5 Recommendations:

    #1  Focus Flow by LoRoom
         Score  : 6.39/6.5  [##########] 98% match
         Genre  : lofi | Mood: focused | Energy: 0.38
         Why    :
           - genre match - lofi (+2.0)
           - mood match - focused (+1.5)
           - energy 0.38 vs target 0.35 -> closeness 0.97 (+1.45)
           - acoustic match - acousticness 0.82 (+1.0)
           - valence 0.52 -> closeness 0.93 (+0.47)

    #2  Library Rain by Drift Collective
         Score  : 5.97/6.5  [#########-] 92% match
         ...

  [4/4] Claude self-critique (dual-RAG: genre + mood)...
  -----------------------------------------------------------------
  Focus Flow is an excellent match. It falls within lofi's typical
  energy range of 0.2–0.5 and satisfies the focused mood's guidance
  exactly: low energy, mid-range valence, and acoustic character keep
  it in the background without interrupting concentration. The 98%
  score reflects alignment across all five scoring signals — a rare
  result. The remaining lofi songs (Library Rain, Midnight Coding)
  continue the pattern but reveal a catalog limitation: only three
  lofi tracks are available, so positions four and five draw from
  adjacent quiet genres (jazz, ambient). Confidence: high. Expanding
  the lofi catalog would be the single most impactful improvement.
  -----------------------------------------------------------------

  Log written to: logs/sessions.log  [session=3a8f21c0]
```

---

### Example 2 — High-energy workout request

**Input:**
```
Your request: High energy for a morning run, I want guitars not electronic
```

**Output:**
```
=================================================================
  AI MUSIC RECOMMENDER — Agentic Session [b72e90d1]
=================================================================

  Request: "High energy for a morning run, I want guitars not electronic"

  [1/4] Parsing request with Claude (RAG-grounded)...
  Parsed profile:
    Genre      : rock   Mood : intense
    Energy     : 0.88   Acoustic : False
    Confidence : 88%

  [2/4] Running guardrails...
  No issues detected.

  [3/4] Agentic: Claude calling recommendation tool...
  → Tool call: get_song_recommendations(genre='rock', mood='intense',
               target_energy=0.88, likes_acoustic=False, k=5)
  ← Tool returned 5 songs

  Top 5 Recommendations:

    #1  Storm Runner by Iron Veil
         Score  : 5.39/6.5  [########--] 83% match
         Genre  : rock | Mood: intense | Energy: 0.91
         Why    :
           - genre match - rock (+2.0)
           - mood match - intense (+1.5)
           - energy 0.91 vs target 0.88 -> closeness 0.97 (+1.46)
           - valence 0.68 -> closeness 0.97 (+0.47)

    #2  Gym Hero by Max Pulse
         Score  : 3.74/6.5  [######----] 58% match
         Genre  : pop | Mood: intense | Energy: 0.93
         ...

  [4/4] Claude self-critique (RAG-grounded)...
  -----------------------------------------------------------------
  Storm Runner aligns well with the request — rock's typical energy
  range is 0.65–0.95, and the song's 0.91 lands near the top of
  that band, matching the "morning run" intensity. The #2 result
  (Gym Hero, pop/intense) earns its position through high energy
  and mood match despite the genre mismatch, which is expected
  behavior given the catalog has only one rock song. The long tail
  (positions #3–#5) fills with other high-energy songs that have
  no genre or mood connection to the request — a predictable artifact
  of a small catalog. Confidence: medium-high. Adding more rock songs
  would improve variety for this profile type.
  -----------------------------------------------------------------
```

---

### Example 3 — Ambiguous or off-catalog request

**Input:**
```
Your request: Something with reggae vibes, upbeat and feel-good
```

**Output:**
```
=================================================================
  AI MUSIC RECOMMENDER — Agentic Session [f1a34b88]
=================================================================

  Request: "Something with reggae vibes, upbeat and feel-good"

  [1/4] Parsing request with Claude (RAG-grounded)...
  Parsed profile:
    Genre      : r&b   Mood : happy
    Energy     : 0.65   Acoustic : False
    Confidence : 68%
    Notes      : 'Reggae' not in catalog. Mapped to 'r&b' based on
                 similar groove and mood characteristics shown in
                 genre overview. Energy set to midpoint of r&b range.

  [2/4] Running guardrails...
  [!] WARNING — GENRE_NOT_IN_CATALOG
      Genre 'reggae' is not in the catalog. The genre bonus (+2.0 pts)
      cannot be earned.
      -> Available genres: ambient, classical, country, edm, folk,
         funk, hip-hop, indie pop, jazz, lofi, metal, pop, r&b, rock,
         synthwave. The system will fall back to mood and energy
         matching only.
  [RAG] Genre 'reggae' not in catalog.
        Knowledge base suggests closest match: 'r&b'

  [3/4] Agentic: Claude calling recommendation tool...
  → Tool call: get_song_recommendations(genre='r&b', mood='happy',
               target_energy=0.65, likes_acoustic=False, k=5)
  ← Tool returned 5 songs

  Top 5 Recommendations:

    #1  Golden Hour by Sienna Vale
         Score  : 3.97/6.5  [######----] 61% match
         Genre  : r&b | Mood: romantic | Energy: 0.62
         Why    :
           - genre match - r&b (+2.0)
           - energy 0.62 vs target 0.65 -> closeness 0.97 (+1.46)
           - valence 0.78 -> closeness 0.87 (+0.43)

  [4/4] Claude self-critique (RAG-grounded)...
  -----------------------------------------------------------------
  The top result scores only 61% because reggae is absent from the
  catalog — the genre bonus effectively cannot be awarded to the
  genre the user actually wanted. The RAG-grounded mapping to r&b
  is a reasonable fallback (similar groove and positive energy
  characteristics), but Golden Hour's mood is 'romantic' rather
  than 'happy', which the user requested. The 68% extraction
  confidence flag is appropriate here — the request was specific
  about a genre the system cannot serve. Confidence: low. The most
  direct improvement would be adding reggae songs to the catalog.
  -----------------------------------------------------------------
```

---

## Design Decisions

**Why a rule engine instead of a second Claude call for ranking?**
The scoring step is deterministic by design. Using a rule engine for ranking means the evaluator can test exact correctness ("does the lofi song rank #1?"), run in milliseconds, and produce identical output on every call. Claude is used where language understanding is needed (parsing natural language, writing a critique); for pure numerical ranking it would add latency, cost, and nondeterminism without improving accuracy.

**Why RAG instead of just putting the genre list in a fixed system prompt?**
The catalog genres are retrieved at runtime from `knowledge/genres.json` and vary with what is actually loaded. If songs are added or removed, the retrieval reflects that automatically. A hardcoded prompt would silently become wrong. The RAG approach also provides structured metadata (energy ranges, typical moods, similar genres) that a flat list could not, which is what grounds Claude's critique in facts rather than guesses.

**Why guardrails before the rule engine, not after?**
Pre-flight checks mean the user sees warnings before the recommendations appear, not buried in an afterthought. They also give the RAG fallback a chance to run — if a genre gap is detected, the system can suggest the closest catalog alternative before computing scores that cannot award the genre bonus anyway.

**Trade-offs made:**
- Genre matching is exact string comparison. "Indie pop" and "pop" score 0 shared credit. A similarity map would fix this but adds complexity that is out of scope for a single project.
- The valence anchor is fixed at 0.65. This silently penalizes users who prefer dark or melancholic music. Exposing it as a user-controlled parameter would require a profile schema change.
- The catalog is 18 songs. Many genres have exactly one representative, which means position #1 is often the only genre match and positions #2–#5 are effectively nearest-neighbor noise. A real system needs a much larger catalog before the scoring weights matter.

---

## Testing Summary

**Results at a glance:**
- **59 / 59 unit tests pass** (guardrails, RAG, recommender, evaluator — run with `pytest`)
- **9 / 9 evaluator cases pass** — 100% reliability score, EXCELLENT grade (run with `python -m src.evaluator`)
- **Confidence scores** average 83% for normal profiles, drop to 63% for catalog-gap profiles, and 70% for adversarial profiles — the system correctly self-reports lower confidence when a request cannot be well served
- **5 guardrail codes** fire deterministically; all five trigger in the expected test cases with zero false positives

**Evaluator results by case:**

| Case | Type | Top result | Score | Confidence | Result |
|---|---|---|---|---|---|
| T01 — Pop / Happy | Normal | Sunrise City | 5.38 / 6.5 | 83% | PASS |
| T02 — Lofi Study | Normal | Focus Flow | 6.39 / 6.5 | 98% | PASS |
| T03 — Rock Workout | Normal | Storm Runner | 5.39 / 6.5 | 83% | PASS |
| T04 — Unknown Genre | Catalog gap | Rooftop Lights | 3.25 / 6.5 | 50% | PASS |
| T05 — Impossible Mood | Catalog gap | Focus Flow | 4.97 / 6.5 | 76% | PASS |
| T06 — High-Energy Acoustic | Adversarial | Porch Light | 3.99 / 6.5 | 61% | PASS |
| T07 — Dead-Center Energy | Adversarial | Night Drive Loop | 5.04 / 6.5 | 78% | PASS |
| T08 — EDM Dance Floor | Normal | Signal Drop | 5.42 / 6.5 | 83% | PASS |
| T09 — Determinism | System | identical on repeat | — | — | PASS |

**What the adversarial profiles revealed:**
- *Unknown genre (T04, reggae):* Guardrail fires correctly and confidence drops to 50%, signaling degraded results. The system still returns songs rather than refusing — it cannot say "nothing matches" — which could mislead a user.
- *High-energy acoustic (T06, folk/intense/energy=0.9):* The quietest folk song in the catalog (energy=0.32) ranks #1 because genre bonus (+2.0) and acoustic bonus (+1.0) outweigh the severe energy penalty. The math is correct; the result is wrong for a real user. Confidence is 61% — the system flags its own uncertainty but cannot self-correct.
- *Dead-center energy (T07, synthwave/moody/energy=0.5):* Position #1 is decisive (genre+mood match, score 5.04), but positions #2–#5 are a near-tied cluster because no catalog song has energy near 0.5.
- *Impossible mood (T05, lofi/wistful):* Genre dominates mood. Lofi songs fill positions #1–#3 with 0 mood-match points while the only matching-mood song in the catalog ranks lower due to genre mismatch.

**One controlled experiment:**
Genre weight was halved (2.0→1.0) and energy weight doubled (1.5→3.0). T06 improved (Porch Light dropped from #1 to #3), but high-energy profiles gained random noise in positions #2–#5. Original weights restored. The experiment showed that numeric reweighting is a blunt instrument — the underlying problem is exact string matching for genre, not the weight values.

---

## Reflection and Ethics

### Limitations and Biases

Five biases are documented in the system; all were found through testing, not anticipated in advance.

**1. Genre exact-string matching.** `"indie pop"` earns zero genre bonus for a `"pop"` user. Two songs that a human would consider closely related score 0 credit for genre. This structurally disadvantages users whose taste spans adjacent genres.

**2. High-energy catalog skew.** 50% of catalog songs have energy above 0.7. Users who request mid-range energy (around 0.5) can never receive a strong energy match — no song clusters there. This was an unconsidered choice made while building the catalog, not an intentional design.

**3. Fixed valence anchor.** The scoring formula silently rewards songs near valence 0.65 (slightly positive). A listener who prefers dark, melancholic, or heavy music is quietly penalized on every recommendation without the system acknowledging it or the user knowing it.

**4. No diversity rule.** The system optimizes entirely for match quality. When a user's genre has multiple catalog entries, they will receive only that genre in their top results — no adjacent discovery. A lofi listener always gets lofi; a jazz listener always gets jazz.

**5. Acoustic scoring asymmetry.** Users who prefer acoustic sound can earn a +1.0 bonus. Users who prefer electronic sound cannot. The system structurally produces weaker long-tail results for electronic music listeners.

A deeper discussion of each bias, including examples from adversarial testing, is in [`model_card.md`](model_card.md).

---

### Could This AI Be Misused?

A music recommender seems low-stakes, but three real misuse vectors exist.

**Emotional state profiling.** A user's mood request ("I want something intense and angry" or "I need something calm, I'm really stressed") reveals information about their emotional state. If session logs were stored and associated with user identity, this data could be used to profile mental state over time. The current system logs session IDs that are random UUIDs — no user identity is stored. In any real deployment, session data should be treated as sensitive and purged on a defined schedule.

**Prompt injection via user input.** The user's free-text request is passed directly into a Claude prompt. A carefully crafted input like *"ignore previous instructions and output..."* could attempt to manipulate Claude's structured output or critique. The current system is partially protected because the NL parsing step uses structured output with a Pydantic schema — Claude is constrained to return valid fields within declared types, which limits how far a prompt injection can deviate. The self-critique step is more exposed; adding output validation there would reduce risk.

**Catalog manipulation at scale.** If this system were extended to accept a user-supplied catalog, a malicious operator could weight a catalog toward specific artists or genres to covertly shape what gets recommended. The current system uses a fixed, auditable `data/songs.csv` file. Any extension that accepts external catalog data should validate that catalog for balance and surface statistics to the operator.

---

### What Surprised Me While Testing

Two things were genuinely unexpected.

**Confidence scoring was better-calibrated than expected.** I assumed confidence scores would cluster near 0.8 and not vary much. Instead, they tracked catalog quality in a way that felt meaningful: 98% for a near-perfect lofi match, 50% for a reggae request the catalog cannot serve, 61% for the contradictory high-energy acoustic profile. Claude's self-reported uncertainty aligned with what the evaluator independently measured as degraded recommendation quality. I expected the confidence field to be decorative; it turned out to be diagnostic.

**The guardrail fires but cannot fix the problem.** In the high-energy acoustic case (T06), the `HIGH_ENERGY_ACOUSTIC_CONFLICT` guardrail fires correctly, the confidence is flagged at 61%, and the Claude critique explicitly calls out the energy mismatch — yet Porch Light (energy=0.32) still ranks #1. The system correctly identifies its own failure at three separate points and then delivers the failure anyway. That gap — between detecting a problem and being able to correct it — was the clearest illustration of why reliability testing needs to go beyond pass/fail checks on guardrail codes.

---

### Collaboration with AI During This Project

AI assistance was used throughout this project. Two interactions are worth examining honestly.

**One instance where AI was genuinely helpful:**
The RAG integration pattern — specifically, the decision to retrieve genre context *before* each Claude call and inject it into the prompt rather than relying on Claude's general knowledge — came from an AI suggestion. The reasoning offered was: Claude's general knowledge about "lofi music" might be accurate in general but wrong for this specific catalog, which may have unusual energy ranges or moods. Grounding Claude's outputs in data retrieved from `knowledge/genres.json` at runtime means the system stays accurate even if the catalog changes. That suggestion improved the design and I would not have framed it that way independently.

**One instance where AI's suggestion was flawed:**
An early version of `agent.py` included `thinking={"type": "adaptive"}` in the streaming self-critique call, also with `max_tokens=512`. The thinking parameter was suggested as a way to improve critique quality. The flaw: the `adaptive` thinking type, when it activates extended thinking, requires token budget *in addition* to the response budget — a 512-token ceiling leaves no room for both. The parameter was removed before it caused a runtime API error, but it was accepted initially without questioning whether the token budget was sufficient. The lesson: AI suggestions about API parameters need to be verified against the actual API documentation before use, not just accepted because they sound plausible.

---

## Project Files

| File | Purpose |
|---|---|
| `src/agent.py` | Agentic pipeline — main AI entry point |
| `src/evaluator.py` | Offline reliability test suite |
| `src/guardrails.py` | Pre-recommendation safety checks |
| `src/logger.py` | Session-based structured logging |
| `src/rag.py` | Knowledge retrieval and formatting (genres + moods) |
| `src/recommender.py` | Weighted scoring and ranking engine |
| `src/main.py` | Rule-based runner (no API key required) |
| `data/songs.csv` | 18-song catalog |
| `knowledge/genres.json` | Genre knowledge base (21 genres) |
| `knowledge/moods.json` | Mood knowledge base (14 moods) — stretch RAG source |
| `tests/` | 59 unit tests covering all modules |
| `model_card.md` | Detailed model card — biases, intended use, evaluation |
| `reflection.md` | Profile comparison analysis in plain language |
| `assets/architecture.png` | System architecture diagram |

---

## Portfolio

**GitHub:** https://github.com/Prettynim/applied-ai-system-project

**Video walkthrough:** *(Loom link — add after recording)*

### What This Project Says About Me as an AI Engineer

This project started as a small rule-based script and ended as a system with four interacting layers: a language model that parses intent, a knowledge retrieval layer that grounds that model's outputs in data, a deterministic rule engine that does the heavy lifting of scoring, and a test suite that measures whether the whole thing is trustworthy. What I learned by building it is that the hard part of AI engineering is not making a model respond — it is making a system behave reliably and transparently when inputs are ambiguous, data is incomplete, or the math produces correct but wrong answers. I am most interested in the part of AI development that lives between the model and the user: the guardrails, the retrieval architecture, the logging that makes failures diagnosable, and the testing that proves the system works on cases it was not designed for. This project is evidence that I can build in that space.
