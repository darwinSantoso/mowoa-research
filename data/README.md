# Data

`wp1_ids.csv` lists the 150 evaluated articles: rows 0–149 of `test.01.jsonl` (fold 1),
in file order. It was written by WP1 (`05_WP1_C1.ipynb`, W1). Columns:

| column | meaning |
|---|---|
| `article` | 1-based article number ("Artikel n" in the thesis) |
| `index_in_test01` | 0-based row in `test.01.jsonl` (the index the Phase B/C/D and WP notebooks use) |
| `indosum_id` | IndoSum `id` field |
| `category`, `source` | IndoSum `category` and `source` fields |

No article text or reference summary is included. Download IndoSum yourself:

- Kaggle mirror used in the runs: `kagglehub.dataset_download("linkgish/indosum")`
- Original: https://github.com/kata-ai/indosum

Then check that the first 150 rows of your `test.01.jsonl` have these IDs, e.g. with
`tools/rebuild_canonical_json.py --indosum <dir>/test.01.jsonl --ids data/wp1_ids.csv`,
which stops on the first mismatch.

If you keep a local copy inside the repository, put it in `data/indosum/`; that folder is
in `.gitignore`.

**Index base.** Row 0 of `test.01.jsonl` is "Artikel 1" in the thesis. The Phase B/C/D and
WP notebooks use the 0-based row index; `results/wp1` and `results/wp2` per-article files
are 1-based (README §2).

**Licence.** IndoSum is licensed under CC BY-SA 4.0. The ID list is derived from it and is
shared under the same licence. Cite Kurniawan & Louvan (2018), see README §2.
