# Writing the Book of Life — LMU SoSe 2026

Short exemplary Book-of-Life pipeline on NLSY79. Predicts self-rated health at
age 40 (5 classes) using Azure. 

## Quickstart

```bash
uv sync # Or pip install -e .
cp .env.example .env       # then fill in API keys
jupyter lab bol_pipeline_simplified.ipynb     # or bol_pipeline.ipynb
```

NLSY79 microdata is not in the repo. Use your (or the provided) tag file and download from the [NLS Investigator](https://www.nlsinfo.org/investigator/).
Drop all extracted files into `data/raw/` — see [Data folder](#data-folder) below.

## Two notebooks

Both produce `data/processed/books.json` of the same shape and feed the
same prediction + metrics cells.

| Notebook | Renderer | When to use |
|---|---|---|
| `bol_pipeline.ipynb` | BOLT framework: DuckDB + recipe YAML + `Paragraph` classes, Recipe-driven config. |
| `bol_pipeline_simplified.ipynb` | Simplified inline approach, without external database. |

Both invoke `reshape_nlsy79.py`, which auto-runs
`data/helpers/build_value_labels.py`.

## Data folder

```
data/
├── raw/                              NLSY79 extracts from NLS Investigator (gitignored except *.NLSY79 tagsets)
├── processed/                        reshape outputs (gitignored)
├── helpers/                          value-label tooling + 1970 Census occupation lookup
├── variable_metadata.csv             curated {qname, readable_name, sentence_template, type, paragraph, comment} per variable
└── variable_metadata_template.csv    auto-emitted by reshape — diff vs the curated file to spot new qnames
```

NLSY79 public files are free to download but the NLS Investigator terms
forbid sharing extracts. Never `git add data/raw/*.csv`;

## Adapting to a new extract

The three NLSY-extract files (`*.csv`, `*.cdb`, `*-value-labels.do`)
share a filename prefix. Override it via `.env`:

```ini
NLSY79_EXTRACT_PREFIX=your_extract_name
```

After dropping new files into `data/raw/`, re-run reshape. Reshape emits
`data/variable_metadata_template.csv` straight after parsing the
codebook; diff against `data/variable_metadata.csv` to spot qnames you
haven't curated yet.

## Adding a new variable

1. **Extract.** Add the RNUM to your NLS Investigator query; re-download
   the three files into `data/raw/`.
2. **Curate metadata.** Add a row to `data/variable_metadata.csv`:
   `qname, readable_name, sentence_template, type, paragraph, comment`.
   - `paragraph` — `"person"` or `"year"`; this drives which output table
     the variable lands in. Leave blank to use the default routing
     (XRND → person, year-tagged → year).
3. **Value-decoding transformation.** Only if raw values need decoding
   (e.g. 2-digit year → 4-digit, feet/inches → cm). Add an entry to
   `TRANSFORMATIONS` in `reshape_nlsy79.py`. Ships today:
   `year_2digit_plus_1900`, `feet_inches_to_cm`, `lbs_to_kg`.
4. **Re-run** the reshape cell.

## Changing the prediction target

Knobs in notebook cell 1.2:

```python
TARGET_RNUM = "H0003400"           # raw RNUM of the outcome
BLOCK_QNAME_PREFIXES = ("h40_",)   # qname prefixes to exclude (leakage guard)
```

Person-vs-year routing is driven by the `paragraph` column in
`data/variable_metadata.csv`, not the notebook.

## What to edit when

| Goal | Edit |
|---|---|
| Add a year-level variable | Re-download `data/raw/`; reshape picks it up automatically. Add a row to `data/variable_metadata.csv` (set `paragraph=year`, optionally fill `sentence_template` for nicer prose). If categorical, edit `data/helpers/value_labels_overrides.csv` if label tweaks are needed. |
| Add a person-level variable | Same, plus set `paragraph=person` for that row. |
| Change date window | `reshape_nlsy79.py` args `drop_year_below` / `drop_year_above` (CSV stage). BOLT also supports `min_spell_year` / `max_spell_year` in `bolt/recipes/template.yaml` (can only narrow further). |
| Change rendered prose | Edit `sentence_template` for the qname in `data/variable_metadata.csv`. |
| Restrict features from appearing in bookss | `features:` whitelist per dataset in `bolt/recipes/template.yaml` (BOLT only), or drop from source data |
| Recode values | New entry in `TRANSFORMATIONS` in `reshape_nlsy79.py`. |
| Fix a wrong label | Add a row to `data/helpers/value_labels_overrides.csv` (`qname, code, label`). Overrides win on conflict. |
| Add another dataset (e.g. siblings, news) | Discuss approach with supervisors. |
| Change prediction target | Notebook cell 1.2 (`TARGET_RNUM`, `BLOCK_QNAME_PREFIXES`). Update `SYSTEM_PROMPT` / SRH-label parsing in §4 cells. |
| Switch LLM backend | `.env`; see [LLM backend](#llm-backend). |

## Overrides

`data/helpers/value_labels.csv` is regenerated each reshape from the
`.do` codebook plus `data/helpers/value_labels_overrides.csv`. If a
label is wrong or missing, edit the overrides file and re-run reshape.
Overrides also disambiguate "same `(qname, code)` → different labels
across waves" conflicts that the build script otherwise drops with a
warning to stdout.

## LLM backend

The notebook autodetects: if `AZURE_OPENAI_ENDPOINT` is set, it uses
Azure AI Foundry's OpenAI-compatible v1 API; otherwise plain OpenAI is also possible.
You could also add further models.

```ini
# .env  (OpenAI)
OPENAI_API_KEY=sk-...

# .env  (Azure AI Foundry)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
AZURE_OPENAI_DEPLOYMENT=<deployment name, e.g. gpt-oss-120b>
```

For Azure, `model=` is the **deployment name**, set via
`AZURE_OPENAI_DEPLOYMENT`.