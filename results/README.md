# Results

Text-free outputs, copied unchanged from the runs (`MyDrive/mowoa_mo_study/`). No file
here contains IndoSum text: only article indices, IDs and numbers.

| File | Written by | Contents |
|---|---|---|
| `phaseD/knee_rouge_seeded.csv` | 03 Phase D, D12 | seeded knee ROUGE-1/2/L per (algorithm, seed), 150 articles |
| `wp1/wp1_summary.txt` | 05 WP1 | run record: Tabel 4.1, Tabel 4.18, paired tables, seed sensitivity |
| `wp1/wp1_per_article.csv` | 05 WP1 | per-article ROUGE of every algorithm (`article` is 1-based) |
| `wp2/wp2_summary.txt` | 06 WP2 | per-stage ROUGE, gap decomposition, Artikel 149, switching weight |
| `wp2/wp2_per_article.csv` | 06 WP2 | per-article stage values (`article` is 1-based) |
| `wp4/wp4_summary.txt` | 07 WP4 | comparator settings, counted vs distinct evaluations, easy articles, runtime |
| `wp4/wp4_per_algorithm.csv` | 07 WP4 | per algorithm and article group (hard / easy / all): attempted vs unique evaluations, share of C(N,3) visited, and the Phase C HV / front-recovery figures |
| `wp5/wp5_summary.txt` | 08 WP5 | main comparison and step-wise ablation tables, consistency |
| `wp5/wp5_paired.csv` | 08 WP5 | paired article-level differences |
| `wp5/wp5_curves.csv` | 08 WP5 | mean HV% vs evaluations per configuration (Gambar 4.8, Tabel 4.14 [E]) |
| `wp5/wp5_tables.csv` | 08 WP5 | indicator means/SDs per configuration (Tabel 4.21 is derived from it) |
| `wp6/wp6_summary.txt` | 09 WP6 | leave-one-out, random sampling, swap search |
| `wp6/wp6_paired.csv` | 09 WP6 | paired article-level differences |
| `wp6/wp6_curves.csv` | 09 WP6 | mean HV% vs evaluations per configuration (evaluations to 99 % HV) |

The per-run tables (`wp2_runs.csv`, `wp4_peer_runs.csv`, `wp5_runs.csv`, `wp6_runs.csv`, …)
are not included; see README §8.

README §6 maps every thesis table and figure to a file and cell. The Drive paths printed
inside the `*.txt` files are part of the run record and were left as they are.

Licence: CC BY-SA 4.0 (derived from IndoSum), see README §10.
