# MOWOA for Indonesian Extractive Summarization (IndoSum)

Code, configuration and text-free results for the Master's thesis
*Multi-Objective Whale Optimization Algorithm (MOWOA) for Extractive Text Summarization*
and the ICOT 2026 paper #224 *"Indonesian News Text Summarization using Multi Objective
Whale Optimization Algorithm"*.

**Authors:** Darwin and Abba Suganda Girsang, Computer Science Department, BINUS Graduate
Program - Master of Computer Science, Bina Nusantara University, Jakarta, Indonesia 11480.

> **No IndoSum text is distributed in this repository.** Only the 150 article IDs, code,
> configuration and numerical results are included. See [Data](#2-data) and
> [What is not included](#8-what-is-not-included-and-how-to-regenerate-it).

---

## 1. Purpose

The thesis studies extractive summarization of Indonesian news as a bi-objective
combinatorial problem and asks two questions:

1. **Optimizer quality.** How well does MOWOA approximate the *exact* Pareto front? Every
   feasible 3-sentence subset (C(N,3) per article) is enumerated, so hypervolume (HV),
   IGD, IGD+, GD, Spacing and front recovery are measured against the true front P\*.
   MOWOA is compared with NSGA-II, MOPSO-CD and MOEA/D, two simple baselines
   (random sampling + archive, 1-swap search), a step-wise ablation and a
   leave-one-out ablation.
2. **Summary quality.** How do the selected summaries score (ROUGE-1/2/L) against the
   IndoSum references, compared with LEAD-3, TextRank, LexRank, a single-objective WOA
   and a fine-tuned IndoBERT extractor, and where is quality lost
   (objective gap, search gap, selection gap)?

**Formulation (k = 3 sentences per summary).**

- Coverage (maximised): f₁(x) = Σ_{i: xᵢ=1} Lᵢ βᵢ, with Lᵢ the LexRank score on the
  BGE-M3 cosine graph (edges where cos > 0.1) and βᵢ the position multiplier
  (sentence 1 → 3.0; 2–3 → 2.5; 4–5 → 1.5; 6–8 → 0.8; ≥ 9 → 0.3).
- Redundancy (minimised): f₂(x) = max(0, mean of the 3 pairwise thresholded-adjacency
  values − 0.2).
- Final selection (*compromise-solution selection*): minimise
  d(x) = √(w_cov·n̂_cov² + w_red·n̂_red²) over the final archive (min–max normalised,
  coverage inverted, zero range → 1, first archive index on ties). Two weight settings are
  reported: (1, 1) and (32, 1).

**MOWOA (final configuration)** = binary WOA with external archive (cap 30) [A],
(μ+λ) elitist survival [E], per-whale guides by crowding-distance tournament [G],
V-shaped transfer |tanh(·)| [V], and up to 10 directed 1-swap local-search probes per
generation that replace offspring slots [LS]. In the code it is
`run_mowoa_final(..., **FROZEN)` with
`FROZEN = dict(transfer='V', guide_mode='per_whale_archive', survival='elitist', use_archive=True, ls_probes=10, ls_mode='directed')`
(`notebooks/02_phaseB_comparison_v4.ipynb`, Cell 13). The code calls this configuration
"MO-WOA+"; the thesis calls it "MOWOA". The code's "MO-WOA" is the naive baseline variant
("MOWOA tanpa mekanisme tambahan").

---

## 2. Data

| Item | Value |
|---|---|
| Dataset | IndoSum (Kurniawan & Louvan, 2018), fold 1 |
| Evaluated set | **first 150 articles of `test.01.jsonl`, in file order** (rows 0–149) |
| Article IDs | [`data/wp1_ids.csv`](data/wp1_ids.csv) (IndoSum `id` field; first ID `1494135000-wanita-terberat-di-dunia-jalani-fisioterapi`) |
| Sentence unit | IndoSum's native sentences (`paragraphs` → `split_text_into_sentences`); no regex re-splitting |
| Hard articles | 33 articles with C(N,3) > 1,480, i.e. N ≥ 22 sentences (C(21,3) = 1,330; C(22,3) = 1,540); the other 117 are "easy" |
| Overlap checks | `train.01` ∩ `test.01` IDs = 0; IndoBERT training articles (`train.01` rows 150–1149) share no text with the evaluated 150 |
| Download used in the runs | Kaggle mirror `linkgish/indosum` via `kagglehub.dataset_download("linkgish/indosum")` |
| Original source | https://github.com/kata-ai/indosum |

**Index conventions.** "Artikel n" in the thesis = row n−1 of `test.01.jsonl`
(notebook 01, cell 43, passes `article_id = idx + 1`). The Phase B/C/D and WP notebooks use the
0-based row index (`article_id = idx`). In `results/wp1/wp1_per_article.csv` and
`results/wp2/wp2_per_article.csv` the `article` column is 1-based (first row = 1). Check
the index base of any other file before joining.

**Licence and attribution of the data.** IndoSum is licensed under the Creative Commons
Attribution-ShareAlike 4.0 International License
(http://creativecommons.org/licenses/by-sa/4.0/). The article ID list in `data/` is
derived from IndoSum and is shared under the same licence. If you use IndoSum, cite:

> Kurniawan, K., & Louvan, S. (2018). IndoSum: A new benchmark dataset for Indonesian
> text summarization. In *2018 International Conference on Asian Language Processing
> (IALP)* (pp. 215–220). IEEE. https://doi.org/10.1109/IALP.2018.8629109

```bibtex
@inproceedings{kurniawan2018,
  title     = {IndoSum: A New Benchmark Dataset for Indonesian Text Summarization},
  author    = {Kurniawan, Kemal and Louvan, Samuel},
  booktitle = {2018 International Conference on Asian Language Processing (IALP)},
  pages     = {215--220},
  year      = {2018},
  address   = {Bandung, Indonesia},
  publisher = {IEEE},
  doi       = {10.1109/IALP.2018.8629109}
}
```

---

## 3. Repository layout

```
mowoa-research/
├── README.md
├── LICENSE                      MIT — code
├── LICENSE-data                 CC BY-SA 4.0 — data/ and results/
├── CITATION.cff
├── requirements.txt             optimizer study + ROUGE (CPU)
├── requirements-gpu.txt         + BGE-M3 embedding and IndoBERT (GPU)
├── .gitignore                   blocks IndoSum text, caches, checkpoints, weights
├── src/
│   └── mowoa_mo_shared.py       shared layer: objectives, repair, enumeration, indicators,
│                                MOWOA / MOPSO-CD / pymoo adapters (identical to the
│                                %%writefile block in Phase B Cell 2, Phase D D2 and
│                                the WP notebooks' module cell)
├── notebooks/                   run in this order (see §5)
│   ├── 01_mowoa_v1_4_5.ipynb               ← [v1_4_5]_thesis_MOWOA+_with_two_objectives_cont.ipynb
│   ├── 02_phaseB_comparison_v4.ipynb       ← MO_comparison_phaseB_v4.ipynb
│   ├── 03_phaseD_analysis.ipynb            ← MO_comparison_phaseD.ipynb
│   ├── 04_eval_v1_4_1_baselines.ipynb      ← [EVAL_v1_4_1]_thesis_evaluation_and_comparison_algorithms.ipynb
│   ├── 05_WP1_C1.ipynb                     ← Thesis_Paper_Revision_WP1_C1_colab_cells.ipynb
│   │                                         native-segmentation baselines; MOWOA under (1,1)
│   ├── 06_WP2_C2.ipynb                     ← Thesis_Paper_Revision_WP2_C2_colab_cells.ipynb
│   │                                         per-stage ROUGE; gap decomposition
│   ├── 07_WP4_C3.ipynb                     ← Thesis_Paper_Revision_WP4_C3_colab_cells.ipynb
│   │                                         evaluation budget; unique evaluations; easy articles
│   ├── 08_WP5_C4.ipynb                     ← Thesis_Paper_Revision_WP5_C4_colab_cells.ipynb
│   │                                         tables + convergence curves from one set of runs
│   └── 09_WP6_C5.ipynb                     ← Thesis_Paper_Revision_WP6_C5_colab_cells.ipynb
│                                             leave-one-out; random sampling; swap search
├── data/
│   ├── README.md
│   └── wp1_ids.csv              the 150 IndoSum IDs (CC BY-SA 4.0)
├── results/                     text-free outputs exactly as produced by the runs
│   ├── README.md
│   ├── phaseD/knee_rouge_seeded.csv
│   ├── wp1/  wp1_summary.txt, wp1_per_article.csv
│   ├── wp2/  wp2_summary.txt, wp2_per_article.csv
│   ├── wp4/  wp4_summary.txt, wp4_per_algorithm.csv
│   ├── wp5/  wp5_summary.txt, wp5_paired.csv, wp5_curves.csv, wp5_tables.csv
│   └── wp6/  wp6_summary.txt, wp6_paired.csv, wp6_curves.csv
└── tools/
    ├── rebuild_canonical_json.py   rebuilds the (not distributed) canonical JSON locally
    ├── check_release.py            pre-publication scan (secrets, links, outputs, text leaks)
    └── same_code_cells.py          proves that stripping a notebook left its code unchanged
```

The numeric prefixes give the run order. The original file names are listed above so
that the thesis, the paper and the revision ledger can still be traced to each file.
Labels used inside the notebooks and result files (R4.5, Q17, B1, D-1, W5, C3, …) are
explained in §12.

---

## 4. Environment

All runs were made on **Google Colab**. Two different Colab images were used:

| Runs | Python | Key versions (as printed in the run logs) |
|---|---|---|
| WP1 (15 Oct 2026) | 3.13.15 | CPU x86_64 × 2; CUDA for IndoBERT inference |
| WP4, WP6 (16–18 Oct 2026) | 3.13.16 | numpy 2.1.3, **pymoo 0.6.2**, Intel Xeon 2.20 GHz, 2 vCPU |
| Notebook 01 (v1.4.4 run log) | 3.12 (`cp312` wheels in the pip log) | numpy 2.0.2, scipy 1.16.3, gensim 4.4.0, Sastrawi 1.0.1, stopwordsiso 0.7.1; T4 GPU |

- **pymoo is pinned to 0.6.2.** Phase C was run with 0.6.2; pymoo 0.6.1.6 gives
  different NSGA-II runs (MOEA/D unchanged). WP4 and WP5 pin 0.6.2 and reproduce the
  Phase C runs 4,500/4,500 per comparator.
- **ROUGE:** Google `rouge-score` 0.1.2, `RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=False)`,
  default tokenizer; per-article F1 averaged over the 150 articles.
- **Embeddings:** `BAAI/bge-m3` via `sentence-transformers`, raw sentences (no
  case folding, stop-word removal or stemming), `normalize_embeddings=True`.
- **IndoBERT:** `indobenchmark/indobert-base-p2`.
- Versions of the other packages were not recorded in the logs; `requirements.txt` leaves
  them unpinned. Before tagging a release, record the full environment of a fresh run
  with `pip freeze > requirements-lock.txt`.

```bash
pip install -r requirements.txt        # Phase D, WP1 (without IndoBERT), WP2, WP4, WP5, WP6
pip install -r requirements-gpu.txt    # adds BGE-M3 (Phase B Cell 4, notebook 01) and IndoBERT
```

The notebooks are written for Colab: they `!pip install` packages, mount Google Drive and
read/write the folder `MyDrive/mowoa_mo_study/` (plus `MyDrive/thesis_mowoa_cache/` for
the IndoBERT files). Create these folders in your own Drive, or edit the path constants
at the top of each notebook if you run elsewhere.

Open the notebooks in Colab (File → Upload notebook) or in Jupyter. Cell 2 of
`02_phaseB_comparison_v4.ipynb` and cell D2 of `03_phaseD_analysis.ipynb` write
`mowoa_mo_shared.py` into the working directory with `%%writefile`; the same module is in
`src/mowoa_mo_shared.py` for use outside the notebooks (copy it to the working directory,
or set `PYTHONPATH=src`). The `!pip` and `%%writefile` lines need Colab or Jupyter.

---

## 5. Run order: what each notebook reads and writes

Paths are relative to `MyDrive/mowoa_mo_study/` unless stated. "Canonical JSON" =
`mowoa_summaries_150_rc1_w32_update_test.json` (Phase B, Phase D) and
`mowoa_summaries_150_rc1_w32.json` (notebook 04) — the same content under two names; it
contains IndoSum text, is **not** distributed, and is rebuilt locally with
`tools/rebuild_canonical_json.py` (§8). Phase B, Phase D and notebook 04 open it by a
relative path, i.e. from the Colab working directory (`/content`); WP1 and WP2 read
`mowoa_summaries_150_rc1_w32_update_test.json` from `MyDrive/mowoa_mo_study/`. Put a copy
in both places.

| # | Notebook | Needs | Reads | Writes | Compute |
|---|---|---|---|---|---|
| 0 | `tools/rebuild_canonical_json.py` | IndoSum download | `test.01.jsonl`, `data/wp1_ids.csv` | both canonical JSON names (local only) | seconds |
| 1 | `01_mowoa_v1_4_5.ipynb` | — (independent) | IndoSum via kagglehub (`train.01`, `test.01`, `dev.01`) | `mowoa_plus_summaries_150_seed0.json` (cell 49, Colab working dir, contains text); `selected_sentences_<id>.tsv` (contains text); figures in `REPORT_FIG_DIR` | T4 GPU; BGE-M3 |
| 2 | `02_phaseB_comparison_v4.ipynb` | canonical JSON (Cells 11, 14) | IndoSum `test.01` | `./mowoa_mo_shared.py` (Cell 2); `article_caches_150.pkl` (Cell 4); `enum_summary.pkl` (Cell 5); `pilot_results.csv` (Cell 8); `ladder_pilot.csv` (Cell 10); `ladder_step4_pilot.csv` (Cell 12); `phaseC/sentences.pkl` (contains text) and `phaseC/article_000.pkl … article_149.pkl` (Cell 13, 22,500 runs); `e2_decomposition.pkl` (Cell 14) | GPU 15–25 min (Cell 4, once); CPU ~4–5 h (Cell 13, checkpointed per article) |
| 3 | `03_phaseD_analysis.ipynb` | step 2, canonical JSON | `article_caches_150.pkl`, `phaseC/*` | `phaseD/knee_rouge.pkl`, `statistics.csv`, `fig_convergence.png`, `fig_pareto_scatter.png`, `ladder/article_XXX.pkl` (D11), `all_runs_scalar.csv`, `article_means_hard.csv`, `knee_rouge_seeded.csv`, `set_coverage_matrix.csv` | CPU; D11 ~1 h |
| 4 | `04_eval_v1_4_1_baselines.ipynb` | canonical JSON; IndoBERT checkpoint (§8) | `thesis_mowoa_cache/indobert_extractor.pt`, `indobert_oracle_labels.pkl` | in `thesis_mowoa_cache/`: `woa_singleobj_summaries_150.json`, `indobert_summaries_150.json`, `all_summaries_150_archive.json` (all contain text); `mowoa_comparison_results.tsv`, `mowoa_per_article_f1_scores.tsv` | GPU; SO-WOA 1–3 h |
| 5 | `05_WP1_C1.ipynb` | steps 2–3 (issued as cells W0–W7 appended after Phase D D1–D3) | IndoSum `test.01`, `train.01`, `README.txt`; `article_caches_150.pkl`; `phaseC/*` incl. `sentences.pkl`; canonical JSON; `thesis_mowoa_cache/indobert_extractor.pt` and oracle-label cache | `wp1/wp1_summary.txt`, `wp1_per_article.csv`, `wp1_ids.csv`, W5 rerun checkpoints | CPU 45–90 min (+ GPU for IndoBERT inference) |
| 6 | `06_WP2_C2.ipynb` | steps 2, 3, 5 | caches; `phaseC/*`; `phaseD/ladder/*`; WP1 outputs; IndoSum references | `wp2/wp2_summary.txt`, `wp2_per_article.csv`, `wp2_runs.csv`, `wp2/figures/fig_artikel149_{id,en}.png`, `fig_stages_{id,en}.png`, `fig_gaps_{id,en}.png` | CPU 2–3 h (27,000 runs) |
| 7 | `07_WP4_C3.ipynb` | steps 2, 6; pymoo 0.6.2 | caches; `phaseC/*`; `wp2/wp2_runs.csv` | `wp4/peer_runs/`, `wp4_summary.txt`, `wp4_per_algorithm.csv`, `wp4_paired_hv.csv`, `wp4_peer_runs.csv`, `wp4_timing.csv` | CPU 2–3 h |
| 8 | `08_WP5_C4.ipynb` | steps 2, 3, 7; pymoo 0.6.2 | caches; `phaseC/*`; `phaseD/ladder/*`; WP4 distinct-subset counts | `wp5/runs/`, `wp5_summary.txt`, `wp5_tables.csv`, `wp5_paired.csv`, `wp5_consistency.csv`, `wp5_curves.csv`, `wp5_runs.csv`, `wp5/figures/fig_convergence_{id,en}`, `fig_convergence_ablation_{id,en}` (PNG 300 dpi + PDF) | CPU 1–2 h (8,910 runs) |
| 9 | `09_WP6_C5.ipynb` | steps 2, 3, 8 | caches; `phaseC/*`; `phaseD/ladder/*` (−LS check); WP5 curves | `wp6/wp6_summary.txt`, `wp6_paired.csv`, `wp6_tables.csv`, `wp6_easy.csv`, `wp6_curves.csv`, `wp6_runs.csv` | CPU 1–1.5 h (14,940 runs) |

Notes on dependencies.

- Steps 1 and 4 are independent of the optimizer study. Step 1 produces the single-run
  Artikel 149 walk-through of Bab 4; step 4 produces the **superseded** regex-segmentation
  rows of the old Tabel 4.18 (v1.4.4) and the IndoBERT model check. The final summary-quality
  numbers come from step 5.
- Phase B Cells 8–12 are the 6-article pilot used to design MOWOA⁺. Cell 13 (Phase C) does
  not depend on them.
- Phase D Cell D4 is a gate: MOWOA⁺'s seeded ROUGE under (32, 1) must be within 0.005 of
  0.6544 (it is 0.6528 ± 0.0005).
- Every WP notebook compares its reruns with the earlier records (Phase C, the D11 ladder,
  WP1/WP2/WP4 values). WP5 (C4) and WP6 (C4) stop on any mismatch, and WP4 stops if its
  pre-check on 3 hard articles (C5) fails. The other checks (WP1 W5, WP2 C5, WP4 C6) print
  the number of matching runs instead of stopping; in the published runs they match in
  every run (see the "reproduction checks" lines in `results/wp1`, `wp2` and `wp4`
  summaries). If you regenerate the article cache yourself, regenerate Phase C from it too
  (see §8, *article cache*).
- The read/write lists for steps 5–9 are taken from the WP notebooks' documented inputs
  and outputs; the exact file names are set in each notebook's first (path) cell.

---

## 6. Reproducing each thesis table and figure

Numbering follows thesis v1.4.5. Paper numbers refer to the ICOT 2026 revised paper.
"(derived)" marks values that were computed from a result file outside the notebooks.

| Thesis item | Paper | Notebook | Cell | Result file |
|---|---|---|---|---|
| Tabel 4.1 — statistics of the evaluated 150 articles | Table II | 05 WP1 | W1 | `wp1_summary.txt` → "TABEL 4.1" |
| Tabel 4.8 — MOWOA configuration | Table I | `src/mowoa_mo_shared.py` constants; 02 Phase B Cell 13 `FROZEN` | — | — |
| Tabel 4.10–4.12, §4.4.2 — single run on Artikel 149, seed 0 | — | 01 v1.4.5 | 29 (`summarize`, `seed=0`), 43 (run loop), 45 (figures) | notebook output (contains text; local only) |
| §4.4.3, Tabel 4.12–4.13 both weights; Gambar 4.9 (Artikel 149 in objective space) | Fig. 3 | 06 WP2 | C8 | `wp2_summary.txt` → "ARTIKEL 149"; `wp2/figures/fig_artikel149_{id,en}.png` |
| Tabel 4.14 [A] (79.65 vs 88.50) | — | 03 Phase D (D5); 07 WP4 | D5; C4 | `wp4_per_algorithm.csv` |
| Tabel 4.14 [E] (84.48 → 79.65) | — | 08 WP5 | C8 | `wp5_curves.csv` |
| Tabel 4.15 — step-wise ablation; paired ablation-step table | Table IV | 08 WP5 (origin: Phase D D11) | C5, C6 | `wp5_summary.txt`, `wp5_paired.csv` |
| Tabel 4.16 — MOWOA vs NSGA-II, MOPSO-CD, MOEA/D; paired MOWOA − comparator table (7 indicators) | Table III | 08 WP5 (origin: Phase D D5) | C5, C6 | `wp5_summary.txt`, `wp5_paired.csv` |
| Tabel 4.17 — set coverage | — | 03 Phase D | D7 | `phaseD/set_coverage_matrix.csv` |
| Comparator settings (§4.5.2) | §III-A | 07 WP4 | C3 | `wp4_summary.txt` → "[C3]" |
| Counted vs distinct evaluations (§4.5.2) | — | 07 WP4 | C7 | `wp4_summary.txt` → "[C7]"; `wp4_per_algorithm.csv` |
| Easy articles (§4.5.2) | — | 07 WP4 (4 algorithms); 09 WP6 (2 baselines) | C4; C6 | `wp4_summary.txt` → "[C4] Q3"; `wp6_summary.txt` |
| Leave-one-out table and paired table (§4.5.1) | — | 09 WP6 | C5 | `wp6_summary.txt` → "[C5]"; `wp6_paired.csv` |
| Two-baseline table and paired values (§4.5.2) | — | 09 WP6 | C6, C7 | `wp6_summary.txt` → "[C6]", "[C7]"; `wp6_paired.csv` |
| Inter-seed / inter-article consistency; non-pilot robustness (§4.5.2) | §IV-A | 08 WP5 | C6, C7 | `wp5_summary.txt` |
| Gambar 4.8 — HV vs evaluations | Fig. 2 | 08 WP5 | C8 | `wp5/figures/fig_convergence_{id,en}`; `wp5_curves.csv` |
| Optional: ablation convergence figure | — | 08 WP5 | C8 | `wp5/figures/fig_convergence_ablation_{id,en}` |
| Evaluations to 99 % HV (175 / 410 / 690) | — | 08 WP5 + 09 WP6 | C8; C7 | `wp5_curves.csv`, `wp6_curves.csv` (derived) |
| Tabel 4.18 — ROUGE-1/2/L of all algorithms (native units) | Table V | 05 WP1 | W6 | `wp1_summary.txt` → "TABEL 4.18"; `wp1_per_article.csv` |
| Paired table MOWOA (1,1) − each comparator | Table V text | 05 WP1 (vs LEAD-3) | W6 | `wp1_summary.txt`; other comparators: `wp1_per_article.csv` (derived) |
| §4.6.1 precision/recall, word counts, identity with LEAD-3 | §IV-C | 05 WP1 | W6 | `wp1_summary.txt` |
| §4.6.2 — post-hoc w_cov table on the test set | §IV-C | 05 WP1 | W6 | `wp1_summary.txt` → "SEED SENSITIVITY" |
| Switching weight w\* (§4.6.2) | — | 06 WP2 | — | `wp2_summary.txt` → "SWITCHING WEIGHT" |
| Tabel 4.19 — best MOWOA (1,1) article (Artikel 72) | — | 05 WP1 | W6 | `wp1_per_article.csv` (scores); text from your local IndoSum copy |
| Tabel 4.20 — retention vs remainder (79.65 → 88.50 → 99.86) | §IV-D | 08 WP5 | C5 | `wp5_summary.txt` |
| Tabel 4.21 — relative range of indicators | — | 08 WP5 | C5 | `wp5_tables.csv` means (derived) |
| §4.7.3 — per-stage ROUGE table; paired step table; gap decomposition | §IV-D | 06 WP2 | — | `wp2_summary.txt`; `wp2_per_article.csv` |
| §4.7.3 figures | — | 06 WP2 | C8 | `wp2/figures/fig_stages_{id,en}.png`, `fig_gaps_{id,en}.png` |
| Seeded (32,1) ROUGE of MOWOA (0.6528 ± 0.0005) | — | 03 Phase D | D4 | `results/phaseD/knee_rouge_seeded.csv` |
| E2 corpus means (oracle 0.8476, best on P\* 0.7305) | §IV-D | 02 Phase B (extended in WP2) | Cell 14 | `e2_decomposition.pkl` (not distributed); `wp2_summary.txt` |
| Runtime per run (paper only) | §IV-A | 07 WP4 | C8 | `wp4_summary.txt` → "[C8]" |
| Superseded: old Tabel 4.18 (regex units) | — | 04 eval | CELL 11, 12b | `mowoa_comparison_results.tsv` |

---

## 7. Seeds and evaluation budget

**Seeds.**

- Every optimizer run (MOWOA, all ablation stages, leave-one-out configurations,
  NSGA-II, MOPSO-CD, MOEA/D, random sampling, swap search) uses seeds **0–29**:
  `np.random.default_rng(seed)` in the shared module, `pymoo.optimize.minimize(..., seed=seed)`
  for NSGA-II and MOEA/D.
- For a given (article, seed) every algorithm starts from the **same initial population**
  P0 (`make_initial_population(cache, seed)`, protocol decision D-1), so comparisons are
  paired.
- MOWOA ROUGE values are per-article means over seeds 0–29; seed 0 is reported beside them.
- The single-run walk-through in notebook 01 uses seed 0. The single-objective WOA uses
  `np.random.seed(42)`. IndoBERT training was not seeded.
- Paired differences are article-level (seed means per article; n = 33 hard articles or
  n = 150), reported as descriptive 95 % t-intervals with a 10,000-resample bootstrap as a
  check. No hypothesis tests or p-values are reported.

**Budget.** **1,480 counted evaluations per run = 30 (initial population) + 50 generations
× 29 offspring.** Slot 1 of the population produces no offspring; the local-search probes
replace up to 10 of the 29 offspring slots and are counted inside the budget.

- Every generated candidate counts as one evaluation, including candidates evaluated
  before; unchanged population members are not re-counted; no algorithm caches
  evaluations. NSGA-II removes duplicates against its population before evaluation.
- MOWOA and all its variants use exactly 1,480 in every run. pymoo stops at a generation
  boundary, so MOPSO-CD and MOEA/D use 1,500; NSGA-II uses a median of 1,492 (max 1,508;
  60 runs on small easy articles end below 1,480). The surplus goes to the comparators.
  Cumulative fronts are truncated to the first 1,480 evaluations for every algorithm.
- Objectives are table look-ups in the per-article enumeration, so an evaluation is cheap;
  BGE-M3 embedding is done once per article (Phase B Cell 4).

---

## 8. What is not included, and how to regenerate it

| Not included | Why | How to regenerate |
|---|---|---|
| IndoSum article text and reference summaries | redistribution of text is avoided; IDs only | Download IndoSum (`kagglehub.dataset_download("linkgish/indosum")` or the original repository); use `test.01.jsonl` rows 0–149 and check them against `data/wp1_ids.csv` |
| Canonical JSON (`mowoa_summaries_150_rc1_w32*.json`): generated summaries, references, source texts | contains text | `python tools/rebuild_canonical_json.py --indosum <dir>/test.01.jsonl --ids data/wp1_ids.csv --out <colab working dir>`, then copy `mowoa_summaries_150_rc1_w32_update_test.json` to `MyDrive/mowoa_mo_study/` for WP1 and WP2 (§5). It rebuilds the three lists from the native sentences: references = IndoSum `summary` joined as in `tokens_to_narrative`; source = all native sentences joined by spaces; generated = the first three native sentences (the w = 32 run delivered native LEAD-3 in 150/150 articles). WP1 W1 then re-checks the JSON against IndoSum (150/150), and eval CELL 12b re-checks the old Tabel 4.18 means |
| Any summary text produced by the code (`mowoa_plus_summaries_150_seed0.json`, `all_summaries_150_archive.json`, `woa_singleobj_summaries_150.json`, `indobert_summaries_150.json`, `selected_sentences_*.tsv`, `phaseC/sentences.pkl`) and all notebook outputs | extracted sentences are IndoSum text | Re-run the notebook that writes them (§5) |
| Article cache `article_caches_150.pkl` (BGE-M3 adjacency, LexRank scores, position bonus), `enum_summary.pkl`, `e2_decomposition.pkl` | caches | Phase B Cells 4, 5, 14. **Caveat:** BGE-M3 embeddings are not guaranteed to be bit-identical across hardware and library versions. Two seed-0 runs of the same MOWOA⁺ (notebook 01 vs the cache pipeline) already differ in 2 of 150 articles under (32, 1) (Artikel 99 and 140). A rebuilt cache can therefore change a few values; every reported optimizer and MOWOA ROUGE number comes from the original cache |
| Run checkpoints and large run tables (`phaseC/article_*.pkl`, `phaseD/ladder/`, `wp1` W5 reruns, `wp2_runs.csv`, `wp4/peer_runs/`, `wp4_peer_runs.csv`, `wp5/runs/`, `wp5_runs.csv`, `wp5_consistency.csv`, `wp6_runs.csv`, `all_runs_scalar.csv`) | size | Re-run steps 2–9 in order; every notebook is checkpointed per article and resumes after an interruption |
| Model weights: `BAAI/bge-m3` | third-party model | Downloaded automatically from the Hugging Face Hub by `sentence-transformers` |
| Model weights: fine-tuned IndoBERT `indobert_extractor.pt`, oracle-label cache `indobert_oracle_labels.pkl` | size; trained on IndoSum | See note below |

**IndoBERT.** The reported IndoBERT rows were produced with a cached checkpoint
(`thesis_mowoa_cache/indobert_extractor.pt`). The same checkpoint reproduces the earlier
regex-unit ROUGE-1 of 0.4660, so it is the model behind the reported numbers; it was
trained on `train.01` rows 150–1149 with greedy ROUGE-L top-3 oracle labels. The code sets
2 epochs, AdamW (lr 2e-5, weight decay 0.01), batch 64 and max 128 tokens, and leaves torch
unseeded; the epoch count and seed of the cached run cannot be confirmed from the
artefacts. In `04_eval_v1_4_1_baselines.ipynb` the training branch refers to `train_df`, which
is only defined in the commented-out CELL 4 block, so training from scratch requires that
block to be re-enabled. A retrained model will not reproduce the reported IndoBERT values
exactly.

---

## 9. Known caveats in the code (documented, not changed)

The code is published as it produced the thesis results. Comments, markdown and some
printed messages were tidied after the runs; `tools/same_code_cells.py` compares a
notebook with its run version and shows that the executed code is unchanged. Some labels
in the code are out of date:

- `04_eval_v1_4_1_baselines.ipynb`: the markdown says "train.01", "v5.1" and
  `mowoa_summaries_150.json`; the code reads the w = 32 JSON built from `test.01`. Its
  CELL 4 re-splits sentences with a regex (16.57 sentences/article instead of the native
  17.49), which is why its baseline rows differ from Tabel 4.18 in thesis v1.4.5. CELL 9's
  single-objective WOA is labelled "ablation" but is a separate scalar-fitness WOA, not an
  ablation of MOWOA. The IndoBERT section's markdown says 2,000 training articles
  (150–2150); the code uses `N_TRAIN_ARTICLES = 1000` (rows 150–1149).
- `KNEE_WEIGHTS = [32, 1]` is the default inside the shared module; the (1, 1) results are
  obtained by re-applying the selection rule to the final archive (WP1 W5).
- Notebook 01 cell 7 computes dataset statistics on a random sample of 150 articles from
  `test.01–05` (seed 42), not on the evaluated set; Tabel 4.1 in thesis v1.4.5 comes from
  WP1 W1.

---

## 10. Licence

- **Code** (`src/`, `notebooks/`, `tools/`): MIT License, see [LICENSE](LICENSE).
- **Data and results** (`data/`, `results/`): CC BY-SA 4.0, see [LICENSE-data](LICENSE-data),
  because the ID list and the per-article scores are derived from IndoSum (CC BY-SA 4.0).

## 11. Citation

See [CITATION.cff](CITATION.cff). Please also cite IndoSum (§2).

## 12. Labels used in the notebooks and result files

The notebooks were written during the thesis and the paper revision, and their headings
and printed output keep the labels of that work. They are kept unchanged so that the
result files in `results/` match a rerun.

| Label | Meaning |
|---|---|
| Cell n (02), D1–D12 (03), W0–W7 (05), C0–C9 (06–09) | cell labels inside each notebook; §5 and §6 refer to them |
| D-1, D-2, D-3, D-4, D-6 | protocol decisions of the comparison, listed in the header of `src/mowoa_mo_shared.py` (D-1 shared initial population, D-2 terminal and cumulative fronts, D-3 budget of 1,480 evaluations, D-4 no tuning, D-6 strata by \|P\*\|). Not the same as the Phase D cell labels D1–D12 |
| Phase A / B / C / D | A = the protocol above; B = `02_phaseB_comparison_v4.ipynb` (set-up, pilot, ablation design); C = the 22,500-run study (Phase B Cell 13); D = `03_phaseD_analysis.ipynb` |
| WP1–WP6, C1–C5 | work packages of the paper revision that needed code (WP1 = C1, WP2 = C2, WP4 = C3, WP5 = C4, WP6 = C5); the WP notebooks print their own number in each result file |
| R1.3, R1.6, R4.5, R4.6, R5.8 | numbers of reviewer comments on the ICOT 2026 submission, as in the response letter; WP4–WP6 answer them |
| B1–B3, Q1–Q28, S9, X4 | items of the author's revision plan (not published); each marks which question a block of output answers, e.g. "Q2 attempted vs unique evaluations" (not every number in the range is used) |
| G1–G7b | correctness gates in Phase B (Cell 7) and the bit-exact check of the ablation runner (Cell 10) |
| [v1_4_1], [v1_4_4], [eval_v1_4_1], cell 20 | earlier versions of the thesis notebook and the cells the shared code was copied from; cell 20 of [v1_4_1] is the original MO-WOA |
| MO-WOA+ / MO-WOA | MOWOA in the thesis / MOWOA without the added mechanisms (§1) |
| "handoff §9.3" (WP4 install line) | internal project notes (not published); the point it refers to, pymoo 0.6.2, is documented in §4 |
| "Journal V2" (WP5 C7) | an earlier manuscript version whose consistency numbers C7 recomputes |
