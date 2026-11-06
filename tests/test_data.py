import pandas as pd
import pytest

from asrs_eval import data

HEADER = (
    " ,Time,Report 1,Report 1,Assessments,Assessments,Events,\n"
    "ACN,Date,Narrative,Synopsis,Primary Problem,Contributing Factors / Situations,Anomaly,\n"
)


def write_export(path, rows):
    path.write_text(HEADER + "".join(",".join(r) + ",\n" for r in rows))


def test_merge_dedupe_and_window(tmp_path):
    write_export(tmp_path / "a.csv", [
        ["1", "202407", "text one", "s", "Aircraft", "x", "y"],
        ["2", "202408", "text two", "s", "Human Factors", "x", "y"],
        ["9", "201501", "stray old", "s", "Weather", "x", "y"],
    ])
    write_export(tmp_path / "b.csv", [
        ["2", "202408", "text two", "s", "Human Factors", "x", "y"],   # duplicate ACN
        ["3", "202409", "", "s", "Weather", "x", "y"],                 # empty narrative
        ["4", "202409", "text four", "s", "", "x", "y"],               # missing label
    ])
    df = data.load_reports(tmp_path, start="2022-01")
    assert df["acn"].tolist() == [1, 2]
    assert df["date"].tolist() == [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-08-01")]


def test_missing_column_is_loud(tmp_path):
    (tmp_path / "bad.csv").write_text(" ,Time\nACN,Date\n1,202407\n")
    with pytest.raises(ValueError, match="missing columns"):
        data.load_reports(tmp_path)


def test_fold_rare_classes():
    s = pd.Series(["A"] * 5 + ["B"] * 3 + ["C"])
    assert data.fold_rare_classes(s, 3).value_counts().to_dict() == {"A": 5, "B": 3, "Other": 1}


def test_cutoff_split_drops_ambiguous_month():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-09-01", "2024-10-01", "2024-11-01"])})
    pre, post = data.cutoff_split(df, "2024-10-15")
    assert pre["date"].tolist() == [pd.Timestamp("2024-09-01")]
    assert post["date"].tolist() == [pd.Timestamp("2024-11-01")]
