"""Data contract and invariant tests validated directly against the real `data/*.csv` files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.storage import (
    QUALY_FILENAME,
    RESULTS_FILENAME,
    SORT_KEYS,
    TIMESERIES_FILENAME,
)
from src.pipeline.transformers import QUALY_COLUMNS, RESULTS_COLUMNS, TIMESERIES_COLUMNS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
WEATHER_FLAGS = ["wet", "dry", "warm", "cold", "windy", "cloudy"]


@pytest.fixture(scope="module")
def datasets():
    df_q = pd.read_csv(DATA_DIR / QUALY_FILENAME, encoding="utf-8", sep=";", low_memory=False)
    df_r = pd.read_csv(DATA_DIR / RESULTS_FILENAME, encoding="utf-8", sep=";", low_memory=False)
    df_t = pd.read_csv(DATA_DIR / TIMESERIES_FILENAME, encoding="utf-8", sep=";", low_memory=False)
    return df_q, df_r, df_t


def test_exact_column_contracts(datasets):
    df_q, df_r, df_t = datasets
    assert list(df_q.columns) == QUALY_COLUMNS
    assert len(df_q.columns) == 22
    assert list(df_r.columns) == RESULTS_COLUMNS
    assert len(df_r.columns) == 29
    assert list(df_t.columns) == TIMESERIES_COLUMNS
    assert len(df_t.columns) == 24


def test_utf8_no_bom_and_lf_line_endings():
    for fname in (QUALY_FILENAME, RESULTS_FILENAME, TIMESERIES_FILENAME):
        raw_head = (DATA_DIR / fname).read_bytes()[:4096]
        assert not raw_head.startswith(b"\xef\xbb\xbf"), f"{fname} must be plain UTF-8 without BOM"
        assert b"\r\n" not in raw_head, f"{fname} must use LF line endings"


def test_zero_duplicate_logical_keys(datasets):
    df_q, df_r, df_t = datasets
    assert not df_q.duplicated(subset=["race_key", "driver_key"]).any()
    assert not df_r.duplicated(subset=["race_key", "driver_key"]).any()
    assert not df_t.duplicated(subset=["race_key", "driver_key", "numero_volta"]).any()


def test_qualy_and_results_one_to_one_key_alignment(datasets):
    df_q, df_r, _ = datasets
    assert len(df_q) == len(df_r)
    q_keys = list(zip(df_q["race_key"], df_q["driver_key"]))
    r_keys = list(zip(df_r["race_key"], df_r["driver_key"]))
    assert q_keys == r_keys


def test_mandatory_non_null_and_binary_flag_domains(datasets):
    df_q, df_r, df_t = datasets

    for df, name in ((df_q, QUALY_FILENAME), (df_r, RESULTS_FILENAME), (df_t, TIMESERIES_FILENAME)):
        for flag in WEATHER_FLAGS:
            assert df[flag].isna().sum() == 0, f"{name}.{flag} contains NULLs"
            assert set(df[flag].unique()).issubset({0, 1}), f"{name}.{flag} outside {{0, 1}}"

    for col in ("posicao_grid",):
        assert df_q[col].isna().sum() == 0

    for col in ("posicao_grid", "posicao_final", "voltas_completadas", "is_overtake", "pit_count_acumulado"):
        assert df_r[col].isna().sum() == 0


def test_formula_invariants_delta_and_overtake(datasets):
    _, df_r, _ = datasets
    expected_delta = df_r["posicao_grid"] - df_r["posicao_final"]
    assert (df_r["delta_posicao_grid_final"] == expected_delta).all()

    expected_overtake = ((df_r["posicao_grid"] > 0) & (df_r["posicao_final"] < df_r["posicao_grid"])).astype(int)
    assert (df_r["is_overtake"] == expected_overtake).all()


def test_full_2025_2026_driver_coverage_and_posicao_volta(datasets):
    df_q, df_r, df_t = datasets

    for year in (2025, 2026):
        q_yr = df_q[df_q["ano"] == year]
        r_yr = df_r[df_r["ano"] == year]
        t_yr = df_t[df_t["ano"] == year]
        assert len(r_yr) > 0, f"No {year} rows found"

        q_avg = q_yr.groupby("race_key")["driver_key"].nunique().mean()
        r_avg = r_yr.groupby("race_key")["driver_key"].nunique().mean()
        t_avg = t_yr.groupby("race_key")["driver_key"].nunique().mean()
        assert q_avg >= 18.0, f"{year} qualy avg drivers/race too low: {q_avg:.2f}"
        assert r_avg >= 18.0, f"{year} results avg drivers/race too low: {r_avg:.2f}"
        assert t_avg >= 18.0, f"{year} timeseries avg drivers/race too low: {t_avg:.2f}"

    # 2023-2026 posicao_volta must be populated (not 100% NULL)
    ts_modern = df_t[df_t["ano"] >= 2023]
    non_null_ratio = ts_modern["posicao_volta"].notna().mean()
    assert non_null_ratio >= 0.95, f"2023-2026 posicao_volta coverage too low: {non_null_ratio:.2%}"


def test_deterministic_sort_order(datasets):
    df_q, df_r, df_t = datasets
    for df, fname in ((df_q, QUALY_FILENAME), (df_r, RESULTS_FILENAME), (df_t, TIMESERIES_FILENAME)):
        sort_cols = SORT_KEYS[fname]
        expected = df.sort_values(by=sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(
            df[sort_cols].reset_index(drop=True),
            expected[sort_cols],
            check_dtype=False,
        )


def test_timeseries_file_size_under_github_limit():
    size_mb = (DATA_DIR / TIMESERIES_FILENAME).stat().st_size / (1024 * 1024)
    print(f"{TIMESERIES_FILENAME} disk size: {size_mb:.2f} MB")
    assert size_mb < 90.0, f"{TIMESERIES_FILENAME} exceeds 90 MB safety threshold ({size_mb:.2f} MB)"
