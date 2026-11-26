"""Atomic CSV storage, upsert, deterministic sorting, and formatting for the 3 analytical views."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from src.pipeline.transformers import QUALY_COLUMNS, RESULTS_COLUMNS, TIMESERIES_COLUMNS

QUALY_FILENAME = "vw_qualy_features.csv"
RESULTS_FILENAME = "vw_resultados_com_clima.csv"
TIMESERIES_FILENAME = "vw_voltas_timeseries.csv"

INT_COLUMNS: Set[str] = {
    "ano",
    "rodada",
    "posicao_grid",
    "posicao_final",
    "voltas_completadas",
    "delta_posicao_grid_final",
    "is_overtake",
    "wet",
    "dry",
    "warm",
    "cold",
    "windy",
    "cloudy",
    "pit_count_acumulado",
    "numero_volta",
    "is_pit_na_volta",
    "voltas_completadas_total",
    "posicao_volta",
    "idade_pneu_volta",
}

FLOAT_PRECISION: Dict[str, int] = {
    "q_best_seconds": 3,
    "q_worst_seconds": 3,
    "q_avg_seconds": 3,
    "diferenca_tempo_segundos": 3,
    "fastest_lap_seconds": 3,
    "pit_total_duration_seconds": 3,
    "pit_mean_duration_seconds": 6,
    "tempo_volta_segundos": 3,
    "duracao_pit_ultimas_1_volta": 3,
    "idade_piloto_corrida": 2,
    "temp_ar_media": 1,
    "temp_pista_media": 1,
    "umidade_media": 1,
}

SORT_KEYS: Dict[str, List[str]] = {
    QUALY_FILENAME: ["ano", "rodada", "race_key", "driver_key"],
    RESULTS_FILENAME: ["ano", "rodada", "race_key", "driver_key"],
    TIMESERIES_FILENAME: ["ano", "rodada", "race_key", "driver_key", "numero_volta"],
}

LOGICAL_KEYS: Dict[str, List[str]] = {
    QUALY_FILENAME: ["race_key", "driver_key"],
    RESULTS_FILENAME: ["race_key", "driver_key"],
    TIMESERIES_FILENAME: ["race_key", "driver_key", "numero_volta"],
}

VIEW_COLUMNS: Dict[str, List[str]] = {
    QUALY_FILENAME: QUALY_COLUMNS,
    RESULTS_FILENAME: RESULTS_COLUMNS,
    TIMESERIES_FILENAME: TIMESERIES_COLUMNS,
}


def read_view_csv(path: Path, columns: List[str]) -> pd.DataFrame:
    """Reads an existing analytical CSV into a typed DataFrame (handles UTF-8 or UTF-8-BOM)."""
    if not path.is_file():
        return pd.DataFrame(columns=columns)

    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        na_values=["NULL", ""],
        keep_default_na=True,
        low_memory=False,
    )
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    return df[columns].copy()


def _format_dataframe_for_csv(df: pd.DataFrame, columns: List[str], sort_cols: List[str]) -> pd.DataFrame:
    """Coerces types, sorts deterministically, and formats strings matching the CSV contract."""
    work = df[columns].copy()

    # Coerce sort keys to proper numeric/string types before deterministic sort
    for col in sort_cols:
        if col in INT_COLUMNS:
            work[col] = pd.to_numeric(work[col], errors="coerce").astype("Int64")
        else:
            work[col] = work[col].astype(str)

    work = work.sort_values(by=sort_cols, ascending=True, kind="mergesort").reset_index(drop=True)

    out = pd.DataFrame(index=work.index)
    for col in columns:
        series = work[col]
        if col in INT_COLUMNS:
            num = pd.to_numeric(series, errors="coerce")
            out[col] = num.map(lambda v: "NULL" if pd.isna(v) else str(int(round(float(v)))))
        elif col in FLOAT_PRECISION:
            prec = FLOAT_PRECISION[col]
            fmt = f"{{:.{prec}f}}"
            num = pd.to_numeric(series, errors="coerce")
            out[col] = num.map(lambda v: "NULL" if pd.isna(v) else fmt.format(float(v)))
        else:
            out[col] = series.map(
                lambda v: "NULL" if (v is None or pd.isna(v) or str(v).strip() == "") else str(v)
            )
    return out


def atomic_write_view_csv(df: pd.DataFrame, path: Path, filename: str) -> None:
    """Deterministically formats, sorts, and atomically writes an analytical CSV (UTF-8, LF, `;`)."""
    columns = VIEW_COLUMNS[filename]
    sort_cols = SORT_KEYS[filename]
    formatted = _format_dataframe_for_csv(df, columns, sort_cols)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            formatted.to_csv(fh, sep=";", index=False, lineterminator="\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def merge_view_data(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    filename: str,
    replace_from_year: Optional[int] = None,
    replace_rounds: Optional[Iterable[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """Idempotently merges new OpenF1 rows into existing historical CSV data."""
    columns = VIEW_COLUMNS[filename]
    logical_keys = LOGICAL_KEYS[filename]

    base = existing_df[columns].copy() if not existing_df.empty else pd.DataFrame(columns=columns)
    if not base.empty:
        base["ano"] = pd.to_numeric(base["ano"], errors="coerce").astype("Int64")
        base["rodada"] = pd.to_numeric(base["rodada"], errors="coerce").astype("Int64")

        if replace_from_year is not None:
            base = base[base["ano"] < int(replace_from_year)].copy()

        if replace_rounds:
            rounds_set = {(int(y), int(r)) for y, r in replace_rounds}
            keep_mask = [
                (int(y), int(r)) not in rounds_set
                for y, r in zip(base["ano"].tolist(), base["rodada"].tolist())
            ]
            base = base.loc[keep_mask].copy()

    frames = [f for f in (base, new_df[columns] if not new_df.empty else pd.DataFrame(columns=columns)) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=columns)

    merged = pd.concat(frames, ignore_index=True)
    for col in ("ano", "rodada"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")
    if "numero_volta" in merged.columns:
        merged["numero_volta"] = pd.to_numeric(merged["numero_volta"], errors="coerce").astype("Int64")

    merged = merged.drop_duplicates(subset=logical_keys, keep="last").reset_index(drop=True)
    return merged[columns]


def get_synced_rounds(data_dir: Path, min_year: int = 2023) -> Set[Tuple[int, int]]:
    """Returns `(ano, rodada)` pairs already present in both qualy and results CSVs."""
    qualy_path = data_dir / QUALY_FILENAME
    results_path = data_dir / RESULTS_FILENAME
    if not qualy_path.is_file() or not results_path.is_file():
        return set()

    df_q = read_view_csv(qualy_path, QUALY_COLUMNS)
    df_r = read_view_csv(results_path, RESULTS_COLUMNS)
    if df_q.empty or df_r.empty:
        return set()

    q_pairs = {
        (int(y), int(r))
        for y, r in zip(df_q["ano"], df_q["rodada"])
        if pd.notna(y) and pd.notna(r) and int(y) >= min_year
    }
    r_pairs = {
        (int(y), int(r))
        for y, r in zip(df_r["ano"], df_r["rodada"])
        if pd.notna(y) and pd.notna(r) and int(y) >= min_year
    }
    return q_pairs & r_pairs
