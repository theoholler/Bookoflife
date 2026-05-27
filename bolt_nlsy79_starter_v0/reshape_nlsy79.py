"""Reshape an NLSY79 raw extract into long person + year tables.

Called from ``bol_pipeline.ipynb``. Almost everything happens inside
``reshape_nlsy79`` itself — read it top-to-bottom. The few helpers below
are extracted only because they're called more than once.

Pipeline:
  1. Parse the .cdb codebook + emit variable_metadata_template.csv;
     read the raw RNUM-headed CSV.
  2. Classify usage of every column → predictors (person / year), target,
     or blocked (e.g., to prevent leakage); build variable_index.
  3. Capture NLSY79 missing-sentinel counts (-1..-5), then replace with NaN.
  4. Build person table; attach outcome_age/outcome_calendar_year if the
     target is measured at a specific age rather than a fixed survey year.
  5. Build year tables (full + spell-pruned).
  6. Run value_labels build + apply per-variable transformations + bake
     categorical labels into the data.
  7. Build targets table.
  8. Write outputs + print descriptive stats.

Outputs (to ``out_dir``):
  - ``nlsy79_person.csv``       one row per respondent, person-scope variables.
  - ``nlsy79_year.csv``         (caseid, survey_year) with spell pruning.
  - ``nlsy79_year_full.csv``    same shape without spell pruning, for inspection/baselines.
  - ``targets.csv``             caseid + target [+ outcome_age + outcome_calendar_year].
  - ``variable_index.csv``      audit table of every codebook variable: routing,
                                codebook free-text (question/comment/note),
                                curated metadata.

Also writes:
  - ``data/variable_metadata_template.csv`` — one row per unique qname
    from the codebook, for diffing against ``data/variable_metadata.csv``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent

VARIABLE_METADATA_CSV = REPO / "data" / "variable_metadata.csv"
VARIABLE_METADATA_TEMPLATE_CSV = REPO / "data" / "variable_metadata_template.csv"
VALUE_LABELS_CSV = REPO / "data" / "helpers" / "value_labels.csv"
BUILD_VALUE_LABELS_SCRIPT = REPO / "data" / "helpers" / "build_value_labels.py"


# ---------------------------------------------------------------------------
# Per-variable value transformations, applied at reshape time.
# Keyed by qname. ``transform`` is a unary callable run on non-null values.
# ``field_label`` (optional) overrides the codebook text in variable_index.csv
# when the transform changes the semantics (e.g. unit conversion).
# ---------------------------------------------------------------------------
TRANSFORMATIONS = {
    # 2-digit codebook year (e.g. 60) → 4-digit (1960).
    "q1_3_a_y": {"transform": lambda v: int(v) + 1900 if int(v) < 100 else int(v)},
    # NLSY packs feet/inches: "505" → 5'05" → 165 cm.
    "health_height": {
        "transform": lambda v: int(round((int(v) // 100) * 30.48 + (int(v) % 100) * 2.54)),
        "field_label": "HEIGHT IN CENTIMETERS",
    },
    # Pounds → kilograms.
    "q11_9": {
        "transform": lambda v: int(round(float(v) * 0.45359237)),
        "field_label": "WEIGHT IN KILOGRAMS",
    },
}


# ---------------------------------------------------------------------------
# Shared helpers (called from multiple places below). All other helpers are
# inlined into ``reshape_nlsy79``.
# ---------------------------------------------------------------------------

def _clean_qname(qname, year=None):
    """Normalise raw qname (lowercase, non-alnum→`_`); strip year suffix so waves collapse."""
    s = re.sub(r"[^0-9A-Za-z]+", "_", qname).strip("_").lower()
    if year is not None:
        for tail in (str(year), str(year)[-2:]):
            if len(s) > len(tail) and s.endswith(tail) and not s[-len(tail) - 1].isdigit():
                s = s[:-len(tail)].rstrip("_")
                break
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _parse_year(year_str):
    """Convert codebook year string to int; None for XRND or unparseable."""
    try:
        return int(year_str)
    except (TypeError, ValueError):
        return None


def _to_int64_if_possible(df):
    """Cast numeric columns of whole-number values to nullable Int64 (avoids '5.0' in CSV output)."""
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        non_null = s.dropna()
        if len(non_null) == 0 or ((non_null - non_null.round()).abs() < 1e-9).all():
            try:
                df[col] = s.astype("Int64")
            except (TypeError, ValueError):
                pass
    return df


def _apply_transformations(df):
    """Apply qname-keyed transformations from ``TRANSFORMATIONS`` in place."""
    for qname, spec in TRANSFORMATIONS.items():
        fn = spec.get("transform")
        if fn is None or qname not in df.columns:
            continue
        df[qname] = df[qname].apply(lambda v, fn=fn: fn(v) if pd.notna(v) else v)


def _bake_categorical_labels(df, value_labels, variable_metadata):
    """Replace raw codes with labels (in place) for type=categorical columns."""
    for qname, meta in variable_metadata.items():
        if meta.get("type") != "categorical" or qname not in df.columns:
            continue
        labels = value_labels.get(qname, {})
        if not labels:
            continue
        df[qname] = df[qname].apply(
            lambda v, labels=labels: labels.get(str(v).removesuffix(".0"), v) if pd.notna(v) else v
        )


def apply_tolerance_dedup(year_df, tolerances):
    """Optional opt-in postprocessor: blank year-table cells whose relative change vs previous obs ≤ tolerance."""
    if not tolerances:
        return year_df
    out = year_df.copy()
    for col, tol in tolerances.items():
        if col not in out.columns or tol <= 0:
            continue
        prev = out.groupby("caseid")[col].transform(lambda s: s.ffill().shift(1))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(out[col] - prev) / np.where(prev == 0, np.nan, prev.abs())
        out.loc[rel.le(tol) & out[col].notna() & prev.notna(), col] = np.nan
    return out


# ---------------------------------------------------------------------------
# Main entry point — called from bol_pipeline.ipynb
# ---------------------------------------------------------------------------

def reshape_nlsy79(
    raw_csv_path,
    cdb_path,
    out_dir,
    *,
    caseid_rnum: str = "R0000100",
    target_rnum: str = "H0003400",
    block_qname_prefixes: Iterable[str],
    block_rnums: Iterable[str] = ("R0410100", "R0410300"),
    drop_year_below: int | None = 1979,
    drop_year_above: int | None = 2026,
    outcome_year_mode: str = "age",
    outcome_age: int = 40,
    birth_year_rnum: str = "R0000500",
):
    """See module docstring for the step-by-step pipeline."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: load codebook + emit metadata template + read raw CSV =======
    # Parse the .cdb into {rnum: {qname, year, question, comment, note}}.
    # Each "block" is delimited by long dashes. We pull:
    #   - the header line:    H00032.00 [SHORTNAME] Survey Year: XRND
    #   - the descriptive label that follows "PRIMARY VARIABLE"
    #   - any "COMMENT:" / "NOTE:" paragraphs (joined with " | " on duplicates).
    codebook = {}
    for block in re.split(r"-{40,}", Path(cdb_path).read_text()):
        m = re.match(r"\s*(\w+)\.(\d+)\s+\[([^\]]+)\]\s+Survey Year:\s*(\S+)", block)
        if not m:
            continue
        prefix, suffix, qname, year = m.groups()
        desc = re.search(
            r"PRIMARY VARIABLE\s*\n\s*\n\s*(.+?)(?:\n\s*\n|\nCOMMENT|\nNOTE)",
            block, re.DOTALL,
        )
        question = desc.group(1).strip().replace("\n", " ") if desc else qname
        # COMMENT/NOTE: split block into blank-separated paragraphs, keep ones starting with those prefixes.
        comments, notes = [], []
        for para in re.split(r"\n\s*\n", block):
            s = para.strip()
            if s.startswith("COMMENT:"):
                comments.append(re.sub(r"\s+", " ", s[len("COMMENT:"):]).strip())
            elif s.startswith("NOTE:"):
                notes.append(re.sub(r"\s+", " ", s[len("NOTE:"):]).strip())
        codebook[prefix + suffix] = {
            "qname": qname, "year": year, "question": question,
            "comment": " | ".join(comments), "note": " | ".join(notes),
        }

    # Emit variable_metadata_template right after parsing the codebook so anyone
    # who just dropped a fresh extract can diff it against their curated
    # data/variable_metadata.csv — without waiting for the full pipeline.
    seen = set()
    template_rows = []
    for rec in codebook.values():
        qn = _clean_qname(rec["qname"], _parse_year(rec["year"]))
        if qn in seen:
            continue
        seen.add(qn)
        template_rows.append({
            "qname": qn, "readable_name": "", "sentence_template": "",
            "type": "", "paragraph": "", "comment": rec.get("comment", ""),
        })
    pd.DataFrame(template_rows).sort_values("qname").to_csv(VARIABLE_METADATA_TEMPLATE_CSV, index=False)

    # Read the raw CSV. Columns are NLS reference numbers (e.g. R0000100, H0003400).
    raw = pd.read_csv(raw_csv_path)
    if caseid_rnum not in raw.columns:
        raise ValueError(f"caseid column {caseid_rnum} not in raw csv")
    raw = raw.rename(columns={caseid_rnum: "caseid"})

    # === Step 2: classify each column → variable_index =======================
    block_prefixes = tuple(p.lower() for p in block_qname_prefixes)
    block_rnums_set = set(block_rnums)

    # Pre-emptive drop: columns whose qname starts with a blocked prefix are
    # removed from `raw` entirely so they never reach the output CSVs.
    raw_dropped_rnums = []
    for col in list(raw.columns):
        if col == "caseid":
            continue
        rec = codebook.get(col)
        if rec is None:
            continue
        qname = _clean_qname(rec["qname"], _parse_year(rec["year"]))
        if any(qname.startswith(p) for p in block_prefixes) and col != target_rnum:
            raw.drop(columns=[col], inplace=True)
            raw_dropped_rnums.append(col)

    # Load curated metadata (qname → readable_name + sentence_template + type + paragraph).
    # `paragraph` ("person" / "year") overrides the default XRND-vs-year-tagged routing.
    # Missing entries default to readable_name=qname, empty templates, and fall back to default routing.
    variable_metadata = {}
    if VARIABLE_METADATA_CSV.exists():
        meta_df = pd.read_csv(VARIABLE_METADATA_CSV, dtype=str, na_filter=False)
        variable_metadata = {
            row["qname"]: {
                "readable_name": row["readable_name"] or row["qname"],
                "sentence_template": row["sentence_template"],
                "type": row["type"],
                "paragraph": row.get("paragraph", ""),
            }
            for _, row in meta_df.iterrows() if row["qname"]
        }

    # One row per codebook RNUM with routing decision + metadata + free-text.
    rows = []
    for rnum, rec in codebook.items():
        if rnum != target_rnum and rnum not in raw.columns and rnum not in raw_dropped_rnums:
            continue
        year = _parse_year(rec["year"])
        qname = _clean_qname(rec["qname"], year)
        meta = variable_metadata.get(qname, {})
        trans = TRANSFORMATIONS.get(qname, {})
        paragraph = meta.get("paragraph", "")
        if rnum == target_rnum:
            table, reason = "target", "target rnum"
        elif rnum in block_rnums_set:
            table, reason = "blocked", "rnum in block_rnums"
        elif rnum in raw_dropped_rnums:
            table, reason = "blocked", "qname prefix blocked"
        elif paragraph == "person":
            table, reason = "person", "variable_metadata: paragraph=person"
        elif paragraph == "year":
            table, reason = "year", "variable_metadata: paragraph=year"
        elif rec["year"] == "XRND":
            table, reason = "person", "XRND"
        elif year is not None \
                and (drop_year_below is None or year >= drop_year_below) \
                and (drop_year_above is None or year <= drop_year_above):
            table, reason = "year", "year-tagged"
        else:
            table, reason = "blocked", f"year {rec['year']} out of range"
        rows.append({
            "rnum": rnum, "qname": qname, "year": year, "table": table, "reason": reason,
            "qname_year": f"{qname}_{year}" if year is not None else qname,
            "field_label": trans.get("field_label") or rec.get("question", ""),
            "readable_name": meta.get("readable_name", qname),
            "sentence_template": meta.get("sentence_template", ""),
            "type": meta.get("type", ""),
            "paragraph": paragraph,
            "comment": rec.get("comment", ""),
            "note": rec.get("note", ""),
        })
    idx = pd.DataFrame(rows)
    idx.to_csv(out_dir / "variable_index.csv", index=False)

    # === Step 3: capture missing-sentinel counts, then -1..-5 → NaN ==========
    # Per readable_name, count how many cells hold each NLSY79 missing code in
    # the raw values. We do this BEFORE replacing them with NaN so the
    # descriptive stats can show *why* a value was missing (refusal vs vskip vs …).
    miss_codes = [-1, -2, -3, -4, -5]
    missing_counts: dict[str, dict[int, int]] = {}
    used_idx = idx[idx["table"].isin(("person", "year", "target"))]
    for _, r in used_idx.iterrows():
        if r["rnum"] not in raw.columns:
            continue
        col = raw[r["rnum"]]
        name = r["readable_name"] or r["qname"]
        bucket = missing_counts.setdefault(name, {c: 0 for c in miss_codes})
        for c in miss_codes:
            bucket[c] += int((col == c).sum())

    body = raw.drop(columns=["caseid"]).where(lambda v: v >= 0, np.nan)
    raw = pd.concat([raw[["caseid"]], body], axis=1)

    # === Step 4: birth-year map + person table ==============================
    birth_year_map = {}
    if birth_year_rnum in raw.columns:
        for cid, by in zip(raw["caseid"], raw[birth_year_rnum]):
            if pd.notna(by):
                birth_year_map[int(cid)] = 1900 + int(by)

    person_rows = idx[idx["table"] == "person"]
    person_df = raw[["caseid"]].copy()
    for _, r in person_rows.iterrows():
        if r["rnum"] not in raw.columns:
            continue
        if r["qname"] in person_df.columns:
            person_df[r["qname"]] = person_df[r["qname"]].combine_first(raw[r["rnum"]])
        else:
            person_df[r["qname"]] = raw[r["rnum"]]

    # In age mode, stamp when the target was measured (per-respondent calendar year).
    if outcome_year_mode == "age":
        person_df["outcome_age"] = outcome_age
        person_df["outcome_calendar_year"] = (
            person_df["caseid"].astype(int).map(birth_year_map) + outcome_age
        )

    # === Step 5: year tables (full + spell-pruned sparse) ===================
    year_rows = idx[idx["table"] == "year"]
    long_frames = [
        pd.DataFrame({
            "caseid": raw["caseid"], "survey_year": r["year"],
            "qname": r["qname"], "value": raw[r["rnum"]].values,
        }).dropna(subset=["value"])
        for _, r in year_rows.iterrows() if r["rnum"] in raw.columns
    ]
    if long_frames:
        long_df = pd.concat(long_frames, ignore_index=True)
        full_df = (long_df.pivot_table(
                       index=["caseid", "survey_year"], columns="qname",
                       values="value", aggfunc="last")
                   .reset_index().sort_values(["caseid", "survey_year"])
                   .reset_index(drop=True))
        full_df.columns.name = None

        # Age-mode cutoff: drop years on or after birth_year + outcome_age (no leakage).
        if outcome_year_mode == "age":
            cutoff = full_df["caseid"].astype(int).map(birth_year_map).fillna(np.inf) + outcome_age
            full_df = full_df.loc[full_df["survey_year"] < cutoff].reset_index(drop=True)

        # Spell pruning: blank a cell equal to the prior non-null observation for that (caseid, qname).
        sparse_df = full_df.copy()
        value_cols = [c for c in sparse_df.columns if c not in ("caseid", "survey_year")]
        for col in value_cols:
            prev = sparse_df.groupby("caseid")[col].transform(lambda s: s.ffill().shift(1))
            sparse_df.loc[(sparse_df[col] == prev) & sparse_df[col].notna(), col] = np.nan
        sparse_df = sparse_df.dropna(subset=value_cols, how="all").reset_index(drop=True)
    else:
        full_df = pd.DataFrame(columns=["caseid", "survey_year"])
        sparse_df = full_df.copy()

    # === Step 6: regenerate value_labels.csv, apply transforms, bake labels ===
    # Run the build script (data/helpers/build_value_labels.py) as a module so
    # any errors surface in this traceback. It rewrites value_labels.csv from
    # the .do file + value_labels_overrides.csv on disk.
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_value_labels", BUILD_VALUE_LABELS_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()

    # Load the freshly-written value_labels.csv into {qname: {code: label}}.
    value_labels: dict[str, dict[str, str]] = {}
    if VALUE_LABELS_CSV.exists():
        vl_df = pd.read_csv(VALUE_LABELS_CSV, dtype=str, na_filter=False)
        for _, r in vl_df.iterrows():
            if r["qname"] and r["code"] and r["label"]:
                value_labels.setdefault(r["qname"], {})[r["code"]] = r["label"]

    # Apply value transformations + bake categorical labels into each table.
    for df in (person_df, full_df, sparse_df):
        _apply_transformations(df)
        _bake_categorical_labels(df, value_labels, variable_metadata)

    # === Step 7: rename qname → readable_name, cast to Int64, write =========
    # Map cleaned qname → readable_name from the variable_index (missing entries keep qname).
    qname_to_readable = {}
    for _, r in idx.iterrows():
        q = r.get("qname")
        if isinstance(q, str) and q:
            qname_to_readable[q] = r.get("readable_name") or q

    for df, name in [
        (person_df, "nlsy79_person.csv"),
        (sparse_df, "nlsy79_year.csv"),
        (full_df, "nlsy79_year_full.csv"),
    ]:
        df.rename(columns=qname_to_readable, inplace=True)
        _to_int64_if_possible(df)
        df.to_csv(out_dir / name, index=False)

    # === Step 8: targets ====================================================
    # The target column is the outcome to predict (e.g. self-rated health at age 40).
    # One row per respondent with a valid target value. In age mode we also stamp
    # outcome_age (when the target was measured, per-run constant) and
    # outcome_calendar_year (= birth_year + outcome_age, per-respondent), so any
    # consumer can join the target back to a survey year. Respondents whose
    # target is missing (already replaced with NaN) are dropped.
    target_series = raw[target_rnum] if target_rnum in raw.columns else pd.Series(np.nan, index=raw.index)
    targets_df = pd.DataFrame({"caseid": raw["caseid"], "target": target_series})
    if outcome_year_mode == "age":
        targets_df["outcome_age"] = outcome_age
        targets_df["outcome_calendar_year"] = (
            targets_df["caseid"].astype(int).map(birth_year_map) + outcome_age
        )
    targets_df = targets_df.dropna(subset=["target"]).reset_index(drop=True)
    _to_int64_if_possible(targets_df)
    targets_df.to_csv(out_dir / "targets.csv", index=False)

    # === Step 9: descriptive stats (one line per variable, to stdout) =======
    # Show N + min/p25/med/p75/max (or unique-count for categoricals) plus a
    # breakdown of NLSY79 missing-sentinel counts captured back in Step 3.
    miss_labels = {-1: "refusal", -2: "dk", -3: "invskip", -4: "vskip", -5: "niv"}

    def _row(name, series):
        vals = series.dropna()
        miss = {miss_labels.get(c, c): n for c, n in sorted((missing_counts.get(name) or {}).items()) if n}
        miss_str = ", ".join(f"{k}:{v}" for k, v in miss.items()) or "—"
        if len(vals) == 0:
            return f"  {name:<27} N=      0  missing=[{miss_str}]"
        if pd.api.types.is_numeric_dtype(vals):
            return (f"  {name:<27} N={len(vals):>6,}  "
                    f"min={vals.min():g}  p25={vals.quantile(.25):g}  "
                    f"med={vals.median():g}  p75={vals.quantile(.75):g}  "
                    f"max={vals.max():g}  missing=[{miss_str}]")
        return f"  {name:<27} N={len(vals):>6,}  unique={vals.nunique()}  missing=[{miss_str}]"

    print("\n=== Person variables ===")
    for c in person_df.columns:
        if c != "caseid":
            print(_row(c, person_df[c]))
    print("\n=== Year variables ===")
    for c in sparse_df.columns:
        if c not in ("caseid", "survey_year"):
            print(_row(c, sparse_df[c]))

    return {
        "person": out_dir / "nlsy79_person.csv",
        "year": out_dir / "nlsy79_year.csv",
        "year_full": out_dir / "nlsy79_year_full.csv",
        "targets": out_dir / "targets.csv",
        "variable_index": out_dir / "variable_index.csv",
    }
