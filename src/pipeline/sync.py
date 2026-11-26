"""CLI entrypoint for the incremental OpenF1 -> CSV synchronization pipeline.

Usage:
    python -m src.pipeline.sync
    python -m src.pipeline.sync --from-year 2023 --cache-dir .cache/openf1
    python -m src.pipeline.sync --year 2026 --round 14 --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.pipeline.dob_resolver import DobResolver
from src.pipeline.openf1_client import OpenF1Client, is_session_completed
from src.pipeline.storage import (
    QUALY_FILENAME,
    RESULTS_FILENAME,
    TIMESERIES_FILENAME,
    atomic_write_view_csv,
    get_synced_rounds,
    merge_view_data,
    read_view_csv,
)
from src.pipeline.transformers import (
    QUALY_COLUMNS,
    RESULTS_COLUMNS,
    TIMESERIES_COLUMNS,
    transform_race,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_DATA_DIR = REPO_ROOT / "data"
MIN_OPENF1_YEAR = 2023
COMPLETE_SEASON_MIN_ROUNDS = 20


def _count_csv_data_rows(path: Path) -> int:
    """Fast line-based data row count without parsing the full CSV into pandas."""
    if not path.is_file():
        return 0
    with open(path, "rb") as fh:
        total_lines = sum(1 for _ in fh)
    return max(0, total_lines - 1)


def select_incremental_years(
    already_synced: Set[Tuple[int, int]],
    current_year: int,
    min_year: int = MIN_OPENF1_YEAR,
    complete_season_min_rounds: int = COMPLETE_SEASON_MIN_ROUNDS,
) -> List[int]:
    """Selects minimal years to inspect in incremental mode.

    Always inspects `current_year`, and includes `current_year - 1` only if it is
    within the OpenF1 window and has fewer than `complete_season_min_rounds` synced.
    """
    prev_year = current_year - 1
    prev_rounds = sum(1 for y, _ in already_synced if y == prev_year)
    if prev_year >= min_year and prev_rounds < complete_season_min_rounds:
        return [prev_year, current_year]
    return [current_year]


def load_configs(config_dir: Path = CONFIG_DIR) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, float]]:
    circuits_map = json.loads((config_dir / "circuits_map.json").read_text(encoding="utf-8"))
    constructors_map = json.loads((config_dir / "constructors_map.json").read_text(encoding="utf-8"))
    climate_thresholds = json.loads((config_dir / "climate_thresholds.json").read_text(encoding="utf-8"))
    return circuits_map, constructors_map, climate_thresholds


def _has_valid_race_results(race_results: List[Dict[str, Any]]) -> bool:
    for r in race_results or []:
        pos = r.get("position")
        if r.get("dns") is True:
            continue
        if isinstance(pos, (int, float)) and not isinstance(pos, bool) and int(pos) >= 1:
            return True
    return False


def _validate_in_memory_contracts(
    df_qualy: pd.DataFrame,
    df_results: pd.DataFrame,
    df_ts: pd.DataFrame,
) -> None:
    """Pre-write sanity checks to prevent writing corrupted data."""
    if list(df_qualy.columns) != QUALY_COLUMNS:
        raise ValueError("vw_qualy_features column contract violation.")
    if list(df_results.columns) != RESULTS_COLUMNS:
        raise ValueError("vw_resultados_com_clima column contract violation.")
    if list(df_ts.columns) != TIMESERIES_COLUMNS:
        raise ValueError("vw_voltas_timeseries column contract violation.")

    if df_qualy.duplicated(subset=["race_key", "driver_key"]).any():
        raise ValueError("Duplicate (race_key, driver_key) detected in vw_qualy_features.")
    if df_results.duplicated(subset=["race_key", "driver_key"]).any():
        raise ValueError("Duplicate (race_key, driver_key) detected in vw_resultados_com_clima.")
    if df_ts.duplicated(subset=["race_key", "driver_key", "numero_volta"]).any():
        raise ValueError("Duplicate (race_key, driver_key, numero_volta) detected in vw_voltas_timeseries.")

    q_keys = set(zip(df_qualy["race_key"], df_qualy["driver_key"]))
    r_keys = set(zip(df_results["race_key"], df_results["driver_key"]))
    if q_keys != r_keys:
        raise ValueError(
            f"1:1 key alignment broken between qualy ({len(q_keys)}) and results ({len(r_keys)})."
        )


def _write_summary_markdown(
    summary_path: Path,
    mode_desc: str,
    processed_rounds: List[Dict[str, Any]],
    before_counts: Dict[str, int],
    after_counts: Dict[str, int],
    file_sizes_mb: Dict[str, float],
    newly_discovered_dobs: Dict[str, str],
    warnings: List[str],
) -> None:
    lines = [
        "## OpenF1 Incremental Data Sync Summary",
        "",
        f"- **Mode:** `{mode_desc}`",
        f"- **Rounds synced:** {len(processed_rounds)}",
        "",
        "### Dataset Row Deltas & Sizes",
        "",
        "| File | Before | After | Delta | Disk Size |",
        "|---|---:|---:|---:|---:|",
    ]
    for fname in (QUALY_FILENAME, RESULTS_FILENAME, TIMESERIES_FILENAME):
        b = before_counts.get(fname, 0)
        a = after_counts.get(fname, 0)
        delta = a - b
        sign = "+" if delta > 0 else ""
        sz = file_sizes_mb.get(fname, 0.0)
        lines.append(f"| `{fname}` | {b:,} | {a:,} | {sign}{delta:,} | {sz:.2f} MB |")

    if processed_rounds:
        lines.extend(["", "### Synced Grands Prix", ""])
        for pr in processed_rounds:
            lines.append(
                f"- `{pr['race_key']}` ({pr['nome_gp']}, {pr['tipo_evento']}) — "
                f"{pr['drivers']} drivers, {pr['laps']} laps"
            )

    if newly_discovered_dobs:
        lines.extend(["", "### Newly Discovered Driver DOBs (`drivers_dob.json`)", ""])
        for drv, dob_str in sorted(newly_discovered_dobs.items()):
            lines.append(f"- `{drv}`: `{dob_str}`")

    if warnings:
        lines.extend(["", "### Warnings", ""])
        for w in warnings:
            lines.append(f"- {w}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_sync(
    data_dir: Path = DEFAULT_DATA_DIR,
    config_dir: Path = CONFIG_DIR,
    from_year: Optional[int] = None,
    year: Optional[int] = None,
    round_num: Optional[int] = None,
    dry_run: bool = False,
    cache_dir: Optional[Path] = None,
    summary_file: Optional[Path] = None,
    now_utc: Optional[datetime] = None,
    client: Optional[OpenF1Client] = None,
    dob_resolver: Optional[DobResolver] = None,
) -> Dict[str, Any]:
    """Executes the OpenF1 -> CSV synchronization pipeline."""
    log = logging.getLogger("openf1_sync")
    ref_now = now_utc or datetime.now(timezone.utc)
    current_year = ref_now.year

    circuits_map, constructors_map, climate_thresholds = load_configs(config_dir)
    api = client or OpenF1Client(cache_dir=cache_dir, logger=log)
    dob = dob_resolver or DobResolver(
        dob_file=config_dir / "drivers_dob.json",
        enable_network_lookup=True,
        persist_updates=not dry_run,
        logger=log,
    )

    existing_qualy = read_view_csv(data_dir / QUALY_FILENAME, QUALY_COLUMNS)
    existing_results = read_view_csv(data_dir / RESULTS_FILENAME, RESULTS_COLUMNS)
    ts_path = data_dir / TIMESERIES_FILENAME

    before_counts = {
        QUALY_FILENAME: len(existing_qualy),
        RESULTS_FILENAME: len(existing_results),
        TIMESERIES_FILENAME: _count_csv_data_rows(ts_path),
    }

    # Determine target years and replacement policy
    replace_from_year: Optional[int] = None
    already_synced: Set[Tuple[int, int]] = set()

    if from_year is not None:
        if from_year < MIN_OPENF1_YEAR:
            raise ValueError(f"--from-year must be >= {MIN_OPENF1_YEAR} (2014-2022 is frozen).")
        target_years = list(range(from_year, current_year + 1))
        replace_from_year = from_year
        mode_desc = f"rebuild from {from_year}..{current_year}"
    elif year is not None:
        if year < MIN_OPENF1_YEAR:
            raise ValueError(f"--year must be >= {MIN_OPENF1_YEAR} (2014-2022 is frozen).")
        target_years = [year]
        mode_desc = f"sync year={year}" + (f" round={round_num}" if round_num else "")
    else:
        if not existing_qualy.empty and not existing_results.empty:
            q_pairs = {
                (int(y), int(r))
                for y, r in zip(existing_qualy["ano"], existing_qualy["rodada"])
                if pd.notna(y) and pd.notna(r) and int(y) >= MIN_OPENF1_YEAR
            }
            r_pairs = {
                (int(y), int(r))
                for y, r in zip(existing_results["ano"], existing_results["rodada"])
                if pd.notna(y) and pd.notna(r) and int(y) >= MIN_OPENF1_YEAR
            }
            already_synced = q_pairs & r_pairs
        else:
            already_synced = get_synced_rounds(data_dir, min_year=MIN_OPENF1_YEAR)
        target_years = select_incremental_years(already_synced, current_year=current_year)
        mode_desc = (
            f"incremental ({target_years[0]}..{target_years[-1]})"
            if len(target_years) > 1
            else f"incremental ({target_years[0]})"
        )

    log.info("Starting OpenF1 sync | mode=%s | dry_run=%s", mode_desc, dry_run)

    new_qualy_frames: List[pd.DataFrame] = []
    new_results_frames: List[pd.DataFrame] = []
    new_ts_frames: List[pd.DataFrame] = []
    replaced_rounds: List[Tuple[int, int]] = []
    processed_rounds: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for yr in target_years:
        meetings = api.discover_gp_meetings(year=yr)
        log.info("Year %d: discovered %d GP meetings.", yr, len(meetings))

        for m in meetings:
            rod = int(m["rodada"])
            if round_num is not None and rod != round_num:
                continue

            # In pure incremental mode, skip rounds already synced
            if from_year is None and year is None and (yr, rod) in already_synced:
                continue

            race_session = m["race_session"]
            if not is_session_completed(race_session, now_utc=ref_now, buffer_hours=2.0):
                log.info(
                    "Skipping future/in-progress GP: %d R%d (%s, date_end=%s)",
                    yr,
                    rod,
                    m["meeting_name"],
                    race_session.get("date_end"),
                )
                continue

            race_sk = int(race_session["session_key"])
            quali_session = m.get("qualifying_session")
            quali_sk = int(quali_session["session_key"]) if quali_session else None

            race_results = api.get_session_result(session_key=race_sk)
            if not _has_valid_race_results(race_results):
                log.info(
                    "Skipping GP without finalized race results: %d R%d (%s, session_key=%d)",
                    yr,
                    rod,
                    m["meeting_name"],
                    race_sk,
                )
                continue

            log.info(
                "Fetching telemetry for %d R%d — %s (race_sk=%d, quali_sk=%s)",
                yr,
                rod,
                m["meeting_name"],
                race_sk,
                quali_sk,
            )

            race_drivers = api.get_drivers(session_key=race_sk)
            race_weather = api.get_weather(session_key=race_sk)
            race_laps = api.get_laps(session_key=race_sk)
            race_positions = api.get_position(session_key=race_sk)
            race_stints = api.get_stints(session_key=race_sk)
            race_pits = api.get_pit(session_key=race_sk)

            quali_drivers: List[Dict[str, Any]] = []
            quali_results: List[Dict[str, Any]] = []
            race_starting_grid: List[Dict[str, Any]] = []
            quali_starting_grid: List[Dict[str, Any]] = []
            quali_positions: List[Dict[str, Any]] = []
            quali_laps: List[Dict[str, Any]] = []

            if quali_sk is not None:
                quali_results = api.get_session_result(session_key=quali_sk)
                # OpenF1 indexes /starting_grid under the Qualifying session_key;
                # query quali_sk first and only fall back to race_sk when empty.
                quali_starting_grid = api.get_starting_grid(session_key=quali_sk)
                if not quali_starting_grid:
                    race_starting_grid = api.get_starting_grid(session_key=race_sk)

                # Only fetch qualifying /drivers if some result driver_number is absent from race /drivers
                race_drv_nums = {
                    int(d["driver_number"])
                    for d in race_drivers
                    if isinstance(d.get("driver_number"), (int, float))
                }
                result_drv_nums = {
                    int(r["driver_number"])
                    for r in race_results
                    if isinstance(r.get("driver_number"), (int, float))
                }
                if result_drv_nums - race_drv_nums:
                    quali_drivers = api.get_drivers(session_key=quali_sk)

                if not race_starting_grid and not quali_starting_grid and not quali_results:
                    quali_positions = api.get_position(session_key=quali_sk)

                # Fetch qualifying /laps only if any driver lacks valid qualifying duration
                q_dur_drivers = set()
                for qr in quali_results:
                    dn = qr.get("driver_number")
                    dur = qr.get("duration")
                    valid_d = (
                        any(isinstance(v, (int, float)) and v > 0 for v in dur)
                        if isinstance(dur, list)
                        else (isinstance(dur, (int, float)) and dur > 0)
                    )
                    if isinstance(dn, (int, float)) and valid_d:
                        q_dur_drivers.add(int(dn))
                if result_drv_nums - q_dur_drivers:
                    quali_laps = api.get_laps(session_key=quali_sk)
            else:
                race_starting_grid = api.get_starting_grid(session_key=race_sk)

            if not race_laps:
                msg = f"{yr} R{rod} ({m['meeting_name']}, session_key={race_sk}): OpenF1 returned 0 laps."
                log.warning(msg)
                warnings.append(msg)

            df_q, df_r, df_t = transform_race(
                meeting_info=m,
                race_drivers=race_drivers,
                quali_drivers=quali_drivers,
                race_results=race_results,
                quali_results=quali_results,
                race_starting_grid=race_starting_grid,
                quali_starting_grid=quali_starting_grid,
                quali_positions=quali_positions,
                quali_laps=quali_laps,
                race_weather=race_weather,
                race_laps=race_laps,
                race_positions=race_positions,
                race_stints=race_stints,
                race_pits=race_pits,
                circuits_map=circuits_map,
                constructors_map=constructors_map,
                climate_thresholds=climate_thresholds,
                dob_lookup=dob.get_dob,
                warnings=warnings,
            )

            if df_r.empty:
                msg = f"{yr} R{rod} ({m['meeting_name']}): 0 transformed race rows; skipping."
                log.warning(msg)
                warnings.append(msg)
                continue

            new_qualy_frames.append(df_q)
            new_results_frames.append(df_r)
            new_ts_frames.append(df_t)
            replaced_rounds.append((yr, rod))
            processed_rounds.append(
                {
                    "ano": yr,
                    "rodada": rod,
                    "race_key": str(df_r["race_key"].iloc[0]),
                    "nome_gp": m["meeting_name"],
                    "tipo_evento": m["tipo_evento"],
                    "drivers": len(df_r),
                    "laps": len(df_t),
                }
            )

    if not processed_rounds and replace_from_year is None:
        log.info("Already up to date: 0 new completed rounds to sync.")
        file_sizes_mb = {
            fname: (data_dir / fname).stat().st_size / (1024 * 1024)
            if (data_dir / fname).is_file()
            else 0.0
            for fname in (QUALY_FILENAME, RESULTS_FILENAME, TIMESERIES_FILENAME)
        }
        if summary_file is not None:
            _write_summary_markdown(
                summary_file,
                mode_desc,
                processed_rounds,
                before_counts,
                before_counts,
                file_sizes_mb,
                dob.newly_discovered,
                warnings,
            )
        return {
            "rounds_synced": 0,
            "before_counts": before_counts,
            "after_counts": before_counts,
            "warnings": warnings,
            "newly_discovered_dobs": dob.newly_discovered,
        }

    new_qualy_all = (
        pd.concat(new_qualy_frames, ignore_index=True)
        if new_qualy_frames
        else pd.DataFrame(columns=QUALY_COLUMNS)
    )
    new_results_all = (
        pd.concat(new_results_frames, ignore_index=True)
        if new_results_frames
        else pd.DataFrame(columns=RESULTS_COLUMNS)
    )
    new_ts_all = (
        pd.concat(new_ts_frames, ignore_index=True)
        if new_ts_frames
        else pd.DataFrame(columns=TIMESERIES_COLUMNS)
    )

    existing_ts = read_view_csv(ts_path, TIMESERIES_COLUMNS)
    before_counts[TIMESERIES_FILENAME] = len(existing_ts)

    merged_qualy = merge_view_data(
        existing_qualy,
        new_qualy_all,
        QUALY_FILENAME,
        replace_from_year=replace_from_year,
        replace_rounds=replaced_rounds,
    )
    merged_results = merge_view_data(
        existing_results,
        new_results_all,
        RESULTS_FILENAME,
        replace_from_year=replace_from_year,
        replace_rounds=replaced_rounds,
    )
    merged_ts = merge_view_data(
        existing_ts,
        new_ts_all,
        TIMESERIES_FILENAME,
        replace_from_year=replace_from_year,
        replace_rounds=replaced_rounds,
    )

    _validate_in_memory_contracts(merged_qualy, merged_results, merged_ts)

    after_counts = {
        QUALY_FILENAME: len(merged_qualy),
        RESULTS_FILENAME: len(merged_results),
        TIMESERIES_FILENAME: len(merged_ts),
    }

    if dry_run:
        log.info(
            "DRY-RUN complete (%d rounds). No files written. After counts: %s",
            len(processed_rounds),
            after_counts,
        )
    else:
        atomic_write_view_csv(merged_qualy, data_dir / QUALY_FILENAME, QUALY_FILENAME)
        atomic_write_view_csv(merged_results, data_dir / RESULTS_FILENAME, RESULTS_FILENAME)
        atomic_write_view_csv(merged_ts, data_dir / TIMESERIES_FILENAME, TIMESERIES_FILENAME)
        log.info(
            "Successfully wrote 3 CSVs (%d rounds synced). Row counts: %s",
            len(processed_rounds),
            after_counts,
        )

    file_sizes_mb = {
        fname: (data_dir / fname).stat().st_size / (1024 * 1024)
        if (data_dir / fname).is_file()
        else 0.0
        for fname in (QUALY_FILENAME, RESULTS_FILENAME, TIMESERIES_FILENAME)
    }

    if summary_file is not None:
        _write_summary_markdown(
            summary_file,
            mode_desc,
            processed_rounds,
            before_counts,
            after_counts,
            file_sizes_mb,
            dob.newly_discovered,
            warnings,
        )

    return {
        "rounds_synced": len(processed_rounds),
        "processed_rounds": processed_rounds,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "file_sizes_mb": file_sizes_mb,
        "warnings": warnings,
        "newly_discovered_dobs": dob.newly_discovered,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.sync",
        description="Incremental, idempotent OpenF1 -> CSV analytical pipeline.",
    )
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Rebuild/replace all rows with ano >= FROM_YEAR from OpenF1 (must be >= 2023).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Sync/replace a single championship year (must be >= 2023).",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Optional round number to sync (requires --year).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run extraction, transformation, and contract checks without modifying files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional local JSON response cache directory (e.g., .cache/openf1).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the 3 analytical CSVs (default: data/).",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Optional path to write a Markdown sync summary (for GitHub Actions PR body).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.round is not None and args.year is None:
        parser.error("--round requires --year.")
    if args.from_year is not None and args.year is not None:
        parser.error("--from-year and --year are mutually exclusive.")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_sync(
        data_dir=args.data_dir,
        from_year=args.from_year,
        year=args.year,
        round_num=args.round,
        dry_run=args.dry_run,
        cache_dir=args.cache_dir,
        summary_file=args.summary_file,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
