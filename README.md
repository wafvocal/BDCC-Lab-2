# BDCC Lab 2: How Readers Arrive at English Wikipedia (2017 to 2026)

Exploratory analysis of the English Wikipedia clickstream on Jojie: how the
mix of arrival routes (search engine, article link, unattributed, other
Wikimedia, external site) changed between November 2017 and March 2026.

## What is in the repository

| Path | What it is |
|---|---|
| `main.ipynb` | The report notebook. Run it top to bottom. |
| `clickroutes/` | Supporting code: Spark session, conversion, route mapping, cohorts, decomposition, checks, figures. Every setting is in `clickroutes/config.py`. |
| `figures/` | Figures saved by the notebook (created on the first run). |

No data is stored in the repository. The notebook reads the raw files from
`/mnt/data/public/wikipedia/clickstream/clickstream/` and writes its
intermediate tables to a private folder, `~/clickstream_work/`, outside the
repository.

## Running it on Jojie

1. Open `main.ipynb` from this folder in Jojie (the notebook imports
   `clickroutes/` from the same folder).
2. Run **Kernel → Restart Kernel and Run All Cells**.
3. The first run converts every monthly file (Section 6.2). Expect a few
   hours; the cell prints one line per batch of months. If the kernel stops,
   run all cells again: finished months are skipped.
4. Later runs skip the conversion and take a few minutes; small result tables
   are cached in the work folder and rebuilt automatically when the data or
   settings change.

The conversion stops by itself if the work folder passes 40 GB (the storage
limit is 50 GB). To free the space after the final run, set
`DELETE_WORK_DIR_AT_END = True` in `clickroutes/config.py`.

## Checking the code style

```bash
python -m pycodestyle clickroutes/*.py
```
