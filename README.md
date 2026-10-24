# TV programme recommendation — midterm

Gleb Alektorov. Final submission: **`final_factorial_v5.csv`**.

| Kaggle score | MAP@5 |
|---|---:|
| Private | 0.44559 |
| Public | 0.45546 |

Evidence: [original Kaggle screenshot](midterm/image.png).

## Deliverables

- [Report (PDF)](solutions/midterm/submission/alektorov_gleb.pdf)
- [Research session](solutions/midterm/submission/session.json): JSON Lines in the homework `.json` packaging.
- [Final Kaggle CSV](solutions/midterm/candidates/final_factorial_v5.csv)
- [Implementation](solutions/midterm/tools/) and [content/ALS models](solutions/midterm/models/)
- [Matched-pool evaluation](solutions/midterm/evidence/matched_pool_comparison.json)
- [Final score evidence](solutions/midterm/evidence/submissions/fifth_submission_screenshot.json)

The final model fuses a categorical identity ranker and a session-replay airing model using equal reciprocal-rank fusion (constant 60). Popularity, content filtering and implicit ALS are compared on chronological holdouts. The report distinguishes original submitted-system scores from the later comparison with identical candidates and fallback.

## Layout and environment

The `solutions/midterm` and `midterm` directories retain the original layout required by relative imports and data paths. Run commands from this repository root. Experiments used Python 3.13.1 on macOS arm64, NumPy 2.5.4, SciPy 1.18.1 and LightGBM 4.7.0. `scikit-learn` is also needed for earlier experiment branches; its version is not pinned here. Stored protocols record model settings, seeds, timing and input hashes.

```sh
python3.13 -m venv solutions/midterm/.venv
solutions/midterm/.venv/bin/python -m pip install -r requirements.txt
solutions/midterm/.venv/bin/python solutions/midterm/tools/verify_metrics.py
```

The metric verification uses 16 hand-calculated fixtures and does not train models. No credential is required to inspect the report, CSV, saved predictions or evidence.

## Data and reproduction

Obtain the five competition files from [Kaggle](https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data) after accepting its rules, and place them in `midterm/data/`. Raw data and large caches are not included. Do not commit API tokens.

The basic preprocessing and controlled content/ALS/popularity comparison can be rebuilt with:

```sh
solutions/midterm/.venv/bin/python solutions/midterm/tools/build_interactions.py
solutions/midterm/.venv/bin/python solutions/midterm/tools/run_round1.py
```

These commands perform computation and create local caches. Existing-cache guards intentionally stop accidental overwriting. The advanced experimental workflow is recorded in `tools/` and the corresponding `evidence/*/protocol.json` files; it is not presented as a tested one-command clean rebuild.

Selected frozen checkpoints are under `checkpoints/`, mirroring their original paths relative to `solutions/midterm/cache/`. They include the final identity/replay pair and the v3/v4 reference models. These checkpoints alone do not replace the required intermediate feature caches. `build_matched_pool_comparison.py` re-evaluates saved score caches once those caches are available; its completed outputs are included in evidence.

## Report regeneration (macOS)

```sh
python3 solutions/midterm/tools/build_report.py
swift solutions/midterm/tools/render_report.swift solutions/midterm/evidence/report_render/report_blocks.json solutions/midterm/submission/alektorov_gleb.pdf solutions/midterm/evidence/report_render
python3 solutions/midterm/tools/build_report_session.py
```

The PDF renderer uses native CoreText/PDFKit and embeds the unchanged Kaggle screenshot. Historical evidence retains original source paths and hashes for provenance; packaging does not claim to reproduce old source hashes after report-link edits. No new training or Kaggle submission was performed while packaging this repository.
