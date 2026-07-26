# AI Lab GUI

A local Streamlit app over the lab. It's a thin, pure-Python layer that *reads the
lab from disk* and *reuses `core/`* — it keeps lessons, experiments, tests, and
scratch work in one place.

## Launch
```bash
cd ai-lab
./start.sh            # background, waits until healthy, prints the URL
./stop.sh             # kills the process and frees the port
./start.sh 8600       # custom port (stop with: ./stop.sh 8600)
```
Or run it in the foreground manually:
```bash
source .venv/bin/activate
pip install -e ".[gui]"          # streamlit + matplotlib + pandas (already installed)
streamlit run gui/app.py
```
Opens at http://localhost:8501. `start.sh` writes a PID to `.gui.pid` and logs to
`.gui.log` (both git-ignored).

## Tracks and screens
- **ANN** — a single neuron → backprop → architectures → attention → LLM practice.
- **ML** — classical machine-learning foundations, supervised/unsupervised methods,
  model selection, practical ML, and worked Python examples.
- **Math** — vectors, calculus, probability, optimization, information theory, and
  numerical computing.
- **Dashboard** — shows the learning path, scans lightweight experiments by default
  (`run.py`; 🟢 done / ⚪ todo / 🔴 error / ⏱️ deferred), and embeds `ROADMAP.md`.
- **Study Coach** — daily loop, oral-exam questions, self-checks, derivations, and
  the checkpoint template.
- **Playgrounds** — interactive neuron, MLP, optimizer, regularization, attention,
  tokenization, and tiny-GPT lessons wired to the lab's reusable concepts.
- **Experiments** — per-experiment Overview (README), Code, and Run (captures stdout).
  The public UI is read-only; personal notes stay in local files, not browser-editable forms.
- **Infobase** — renders the `infobase/*.md` notes.
- **Tests** — local-only; runs `pytest -q` and shows passed/skipped/failed. Enabled by
  `./start.sh` through `AILAB_ENABLE_TESTS=1`.
- **Sandbox** — local-only Python scratchpad, enabled by `./start.sh` through
  `AILAB_ENABLE_SANDBOX=1`.

## Languages
- Sidebar language selector: English / Türkçe.
- Nav titles, section headers and per-page chrome live in `gui/i18n.py` (the `TEXT` catalog).
- **Page body text** (lesson prose, widget labels, metrics) stays English in the view files —
  the single source of truth. `gui/catalog_tr.py` maps each English UI string to Turkish, and
  `i18n.install_localization()` wraps Streamlit's text/widget helpers to swap them in when
  Türkçe is selected. Anything not in the catalog (dynamic f-strings, SVG diagram labels)
  falls back to English, so the app always renders.
- Long Markdown content stays as Markdown. Add Turkish files beside English originals:
  `README.tr.md`, `tier2_training.tr.md`, `sigmoid-derivative.tr.md`, etc. (read via
  `read_localized`). If a Turkish file is missing, the app falls back to the English original.

## Layout
```
gui/
├── app.py            # entrypoint: st.navigation over the views
├── lab.py            # disk discovery + run helpers (the only "model" layer)
├── ui.py             # small shared visual polish helpers
└── views/            # one file per screen
    ├── dashboard.py
    ├── study_coach.py
    ├── ml_*.py
    ├── math_*.py
    ├── playground.py
    ├── experiments.py
    ├── infobase.py
    └── tests.py
```

## Extending
Adding an experiment under `experiments/tierN_*/eNN_*/` makes it appear
automatically (Dashboard, Experiments). New `infobase/*.md` notes appear in the
Infobase screen with no code change.
