# AI Lab

A personal lab for learning **artificial neural networks and LLMs from scratch** —
starting from a single artificial neuron and building up, one experiment at a time,
to a small GPT.

## Design principle: one library, many experiments

There is **one growing library** (`core/`) of reusable building blocks. Every
experiment under `experiments/` only *orchestrates and observes* — it imports from
`core/` and never reimplements a neuron, a layer, an optimizer, etc. A neuron you
write in Tier 0 is literally the same class that, many layers deep, becomes a
Transformer's MLP block.

```
core/            the building blocks (grows each tier)
experiments/     the hierarchy of studies (tier0 → tier5)
infobase/        your knowledge base, written in your own words
papers/          annotated reading list + PDFs
datasets/        small datasets
tests/           gradient checks etc. — proves core/ is correct
```

## Setup

```bash
cd ai-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[viz,dev]"      # numpy + matplotlib + pytest
```

> **Python version note:** Tiers 0–1 (the "from scratch" work) need only NumPy and
> run on any modern Python, including 3.14. For Tiers 3+ (PyTorch) you may need a
> separate Python 3.12 venv if PyTorch wheels lag the newest interpreter. See
> `requirements.txt`.

## Run your first experiment

```bash
python experiments/tier0_neuron/e01_single_neuron/run.py
```

Then read [`experiments/tier0_neuron/e01_single_neuron/README.md`](experiments/tier0_neuron/e01_single_neuron/README.md)
— it's the template for every experiment that follows.

## Or launch the GUI

A local Streamlit dashboard ties it all together — ANN, classical ML, and math
tracks; interactive playgrounds; experiment runner; infobase reader; roadmap
progress; and test runner:

```bash
pip install -e ".[gui]"
./start.sh
```

See [`gui/README.md`](gui/README.md).

## Languages

The GUI supports English and Turkish. UI labels live in `gui/i18n.py`; longer
Markdown content can add a Turkish sibling file such as `README.tr.md`,
`checkpoint.tr.md`, or `adam-update.tr.md`. When Turkish is selected, the app
uses the `.tr.md` file when present and falls back to English otherwise.

## How to use this lab

1. Read the matching `infobase/` note for the concept.
2. Open the experiment's `README.md` — it states a **hypothesis** and **what to observe**.
3. Implement / run `run.py`, record results in the experiment's `notes.md`.
4. When you build something reusable, promote it into `core/` and add a test.
5. Use `learning/templates/checkpoint.md` and `learning/questions/` to prove you
   can reconstruct the idea from memory.

See [`ROADMAP.md`](ROADMAP.md) for the full plan and your progress checklist.


## Guided learning (v1.02)

Open **Start here** for a bilingual 20-minute introduction, then save a checkpoint
on any lesson: explored, explained, or rebuilt. **My learning** contains bookmarks,
notes, a comparison notebook, and JSON export/import. Records persist in your browser's
local storage and are processed during your app session; no learner database is created.
Use export before clearing browser data or changing devices.

**Study Coach** suggests your next topic and a recall exercise from your saved progress.
**Debugging challenges** cover divergence, leakage, saturation and wrong gradients.
**Evaluation challenges** cover imbalance, retrieval misses and insufficient evidence.
**Build the bridges** connects NumPy/PyTorch MNIST (e12/e13) to attention, multiple
heads and a Transformer block (e18–e20). Heavy experiments run locally; the public
server never downloads MNIST or launches PyTorch jobs from a visitor request.

```bash
pip install -e '.[gui,dev]'
python -m pytest -q
python -m pytest -q -m slow tests/test_pages_render.py
```

The GUI requires Streamlit 1.58+ and scikit-learn. The fast test suite covers learning
imports and mathematical invariants; the slow suite renders every view. Check shared
links in a fresh browser session as part of each release.

## What's new and future updates

The seven **Guided start** lessons (dot products, gradients, regression,
classification, probability, optimization and model evaluation) connect worked examples
to prediction-first exploration and 56 checked
exercises. Study Coach follows the selected ANN, ML or Math track. Guided exercise
results are saved separately from self-assessed checkpoints, retain hint/reveal use,
and travel with the browser learning export. They are practice records, not grades.

About includes weekly direct page loads, the total since 1 October 2026 and a daily
average over recorded, completed UTC days. The app reads only an aggregate snapshot;
the daily server job and its counting/coverage rules are documented in
[`ops/public-statistics.md`](ops/public-statistics.md).

The homepage shows the latest release on **Welcome**, with a dedicated **What's new**
tab for the full history. Maintain `gui/content/releases.json` for every update:
prepend a higher version, release date, title and learner-facing changes (including
useful lesson links). Preserve published entries. The displayed app version comes
from this same file. When explicitly keeping the public version, prepend a higher
integer `revision` instead (the first entry for a version defaults to revision 1).
What's new groups the dated entries for display; 20 September 2026 appears once as
v1.13. Run `python tools/check_release_notes.py` before deploying;
`tools/release_update.sh` also compares against the live release and refuses updates
without a new entry.
