# AI Lab 1.02 — learner workflow

Delivered: globally registered public lesson routes; a bilingual first session; lesson
objectives/prerequisites/next links; browser-saved checkpoints and bookmarks; portable
notebook and comparison; progress-aware Study Coach; prediction-first exercises with
hints; debugging and evaluation challenges; MNIST NumPy/PyTorch bridge (e12/e13);
NumPy attention/multihead/block experiments (e18/e19/e20); infobase concept maps;
lesson correction links; conditional activation tabs and bounded simulation caching.

Learning data is browser-local persistence with transient server processing. It is not
anonymous server analytics or an account system. Export/import is capped at 500 KB and
100 runs. Unknown topic IDs are ignored; invalid versions and malformed files rejected.
Cross-tab saves merge checkpoints by update date and notebook runs by unique ID.

Human translation review and independent scientific review are not claimed. The guided
introduction has complete EN/TR text; other Turkish lessons remain a partial preview.
Experiment e14–e17 and later training extensions remain future curriculum work.

## Validation

Fast tests cover finite-difference gradients, NumPy/PyTorch update parity (when torch is
installed), causal masking, residual shapes, challenge outcomes and progress-file validation.
The full slow suite checks that all pages render and produce content. Browser checks cover
fresh-session links, checkpoint persistence after reload, and the learning notebook.

A local MNIST run on 3,000 training / 500 test images for four epochs achieved 82.6%
accuracy versus the 13.4% training-majority baseline; matched PyTorch parameters differed
by at most 3.4e-16. This is a small teaching subset, not a full MNIST benchmark.

## Existing-server release

`tools/release_update.sh` stages a versioned directory, installs pinned direct dependencies,
records the resolved dependency lock, and runs tests before allowing activation. It preserves
the deployed private operations code and shared feedback directory. Activation atomically
changes `/opt/ai-lab/current` and restarts only `ai-lab` and `ai-lab-ops`. Failed health checks
restore the previous symlink. Nginx, LMS services and data are not part of this deployment.

The root README's editable GUI install now declares scikit-learn, scipy, threadpoolctl and
the Streamlit minimum needed for lazy tabs and browser storage.
