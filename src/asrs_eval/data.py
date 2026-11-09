"""Load, clean, and split ASRS reports by model training cutoff.

ASRS Database Online exports are CSVs with a two-row header (a group row such as
"Assessments" and a field row such as "Primary Problem"). Exports are capped at 5,000
rows, so several date-window exports are merged and de-duplicated on ACN.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# (group, field) in the export header -> our column name
COLUMNS = {
    ("", "ACN"): "acn",
    ("Time", "Date"): "date",
    ("Report 1", "Narrative"): "narrative",
    ("Report 1", "Synopsis"): "synopsis",
    ("Assessments", "Primary Problem"): "primary_problem",
    ("Assessments", "Contributing Factors / Situations"): "contributing_factors",
    ("Events", "Anomaly"): "anomaly",
}

OTHER = "Other"


def read_export(path: str | Path) -> pd.DataFrame:
    """Read one ASRS export CSV and return the columns we use, with clean names."""
    raw = pd.read_csv(path, header=[0, 1], low_memory=False, dtype=str)
    raw.columns = [(str(a).strip(), str(b).strip()) for a, b in raw.columns]
    missing = [k for k in COLUMNS if k not in raw.columns]
    if missing:
        raise ValueError(f"{path}: export is missing columns {missing}")
    df = raw[list(COLUMNS)].copy()
    df.columns = list(COLUMNS.values())
    df["source_file"] = Path(path).name
    return df


def load_reports(raw_dir: str | Path, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Merge every export in raw_dir, de-duplicate, clean, and filter to [start, end] (YYYY-MM)."""
    files = sorted(Path(raw_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV exports in {raw_dir}")
    df = pd.concat([read_export(f) for f in files], ignore_index=True)
    return clean(df, start=start, end=end)


def clean(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df["acn"] = pd.to_numeric(df["acn"], errors="coerce").astype("Int64")
    # ASRS dates are month-level, YYYYMM
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m", errors="coerce")
    for c in ["narrative", "synopsis", "primary_problem"]:
        df[c] = df[c].astype("string").str.strip()
    df = df.dropna(subset=["acn", "date", "narrative", "primary_problem"])
    df = df[df["narrative"].str.len() > 0]
    df = df.drop_duplicates(subset="acn").sort_values(["date", "acn"]).reset_index(drop=True)
    # Exports occasionally include stray reports dated far outside the searched window.
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def fold_rare_classes(labels: pd.Series, min_count: int) -> pd.Series:
    """Map classes with fewer than min_count examples to OTHER."""
    counts = labels.value_counts()
    rare = counts[counts < min_count].index
    return labels.where(~labels.isin(rare), OTHER)


def add_label(df: pd.DataFrame, min_class_count: int) -> pd.DataFrame:
    df = df.copy()
    df["label"] = fold_rare_classes(df["primary_problem"], min_class_count)
    return df


def train_test(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split. Train feeds the TF-IDF baseline and the retrieval pool;
    test is the only pool LLM evaluation items are drawn from, so nothing leaks."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


PROCESSED_COLUMNS = ["acn", "date", "split", "primary_problem", "label", "synopsis", "narrative"]


def load_processed(path: str | Path) -> pd.DataFrame:
    """Read the frozen dataset written by scripts/build_dataset.py."""
    df = pd.read_csv(path, dtype={"acn": int, "date": str}, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m")
    return df


def cutoff_split(df: pd.DataFrame, cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into reports whose incident month ends before the cutoff, and those after it.

    ASRS dates are month-level, so the month containing the cutoff is ambiguous and is
    dropped from both sides rather than guessed.
    """
    cut = pd.Timestamp(cutoff)
    month = cut.to_period("M").to_timestamp()
    pre = df[df["date"] < month]
    post = df[df["date"] > month]
    return pre.reset_index(drop=True), post.reset_index(drop=True)
