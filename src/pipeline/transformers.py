"""Pure transformation functions: OpenF1 JSON payloads -> 3 canonical analytical DataFrames."""

from __future__ import annotations

from bisect import bisect_right
from datetime import date, datetime, timedelta
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import unicodedata

import numpy as np
import pandas as pd

from src.pipeline.openf1_client import parse_iso_utc

logger = logging.getLogger(__name__)

QUALY_COLUMNS: List[str] = [
    "race_key",
    "driver_key",
    "constructor_key",
    "circuit_key",
    "ano",
    "rodada",
    "nome_gp",
    "tipo_evento",
    "posicao_grid",
    "q_best_seconds",
    "q_worst_seconds",
    "q_avg_seconds",
    "idade_piloto_corrida",
    "wet",
    "dry",
    "warm",
    "cold",
    "windy",
    "cloudy",
    "temp_ar_media",
    "temp_pista_media",
    "umidade_media",
]

RESULTS_COLUMNS: List[str] = [
    "race_key",
    "driver_key",
    "constructor_key",
    "circuit_key",
    "ano",
    "rodada",
    "nome_gp",
    "tipo_evento",
    "posicao_grid",
    "posicao_final",
    "voltas_completadas",
    "delta_posicao_grid_final",
    "is_overtake",
    "diferenca_tempo_segundos",
    "fastest_lap_seconds",
    "idade_piloto_corrida",
    "wet",
    "dry",
    "warm",
    "cold",
    "windy",
    "cloudy",
    "temp_ar_media",
    "temp_pista_media",
    "umidade_media",
    "status_final",
    "pit_count_acumulado",
    "pit_total_duration_seconds",
    "pit_mean_duration_seconds",
]

TIMESERIES_COLUMNS: List[str] = [
    "race_key",
    "driver_key",
    "constructor_key",
    "circuit_key",
    "ano",
    "rodada",
    "tipo_evento",
    "numero_volta",
    "tempo_volta_segundos",
    "posicao_volta",
    "composto_atual",
    "idade_pneu_volta",
    "is_pit_na_volta",
    "duracao_pit_ultimas_1_volta",
    "wet",
    "dry",
    "warm",
    "cold",
    "windy",
    "cloudy",
    "temp_ar_media",
    "temp_pista_media",
    "umidade_media",
    "voltas_completadas_total",
]

_VALID_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"})


def _is_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and np.isfinite(val)


def normalize_slug(raw: Optional[str]) -> str:
    """NFKD accent-strip -> lowercase -> non-alphanumeric to underscore."""
    if not raw or not str(raw).strip():
        return ""
    ascii_str = unicodedata.normalize("NFKD", str(raw).strip()).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_str.lower()).strip("_")


def canonicalize_circuit(
    raw_name: Optional[str],
    circuits_map: Dict[str, str],
    warnings: Optional[List[str]] = None,
) -> str:
    """Resolves an OpenF1 circuit short name to its canonical `circuit_key`."""
    slug = normalize_slug(raw_name)
    if not slug:
        return "unknown"
    if slug in circuits_map:
        return circuits_map[slug]
    msg = f"Unmapped circuit '{raw_name}' (normalized='{slug}'); using normalized key."
    logger.warning(msg)
    if warnings is not None and msg not in warnings:
        warnings.append(msg)
    return slug


def canonicalize_constructor(
    raw_name: Optional[str],
    constructors_map: Dict[str, str],
    warnings: Optional[List[str]] = None,
) -> str:
    """Resolves an OpenF1 team name to its canonical lineage `constructor_key`."""
    if not raw_name or not str(raw_name).strip():
        return "Unknown"
    slug = normalize_slug(raw_name)
    if slug in constructors_map:
        return constructors_map[slug]
    clean = str(raw_name).strip()
    msg = f"Unmapped constructor '{clean}' (normalized='{slug}'); preserving raw name."
    logger.warning(msg)
    if warnings is not None and msg not in warnings:
        warnings.append(msg)
    return clean


def compute_driver_age(
    race_date: Optional[date | str],
    dob: Optional[date | str],
) -> Optional[float]:
    """Calculates fractional driver age on race day: round((race_date - dob).days / 365.25, 2)."""
    if race_date is None or dob is None:
        return None
    try:
        r_dt = date.fromisoformat(str(race_date)[:10]) if isinstance(race_date, str) else race_date
        d_dt = date.fromisoformat(str(dob)[:10]) if isinstance(dob, str) else dob
        return round((r_dt - d_dt).days / 365.25, 2)
    except ValueError:
        return None


def aggregate_weather(
    weather_records: List[Dict[str, Any]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Aggregates OpenF1 `/weather` records into the 9 canonical meteorological columns."""
    warm_min = float(thresholds["warm_min_celsius"])
    cold_max = float(thresholds["cold_max_celsius"])
    windy_min = float(thresholds["windy_min_kmh"])

    if not weather_records:
        return {
            "wet": 0,
            "dry": 1,
            "warm": 0,
            "cold": 0,
            "windy": 0,
            "cloudy": 0,
            "temp_ar_media": None,
            "temp_pista_media": None,
            "umidade_media": None,
        }

    air_temps = [float(r["air_temperature"]) for r in weather_records if _is_number(r.get("air_temperature"))]
    track_temps = [float(r["track_temperature"]) for r in weather_records if _is_number(r.get("track_temperature"))]
    humidities = [float(r["humidity"]) for r in weather_records if _is_number(r.get("humidity"))]
    winds_kmh = [float(r["wind_speed"]) * 3.6 for r in weather_records if _is_number(r.get("wind_speed"))]
    rainfalls = [float(r["rainfall"]) for r in weather_records if _is_number(r.get("rainfall"))]

    mean_air_raw = float(np.mean(air_temps)) if air_temps else None
    temp_ar_media = round(mean_air_raw, 1) if mean_air_raw is not None else None
    temp_pista_media = round(float(np.mean(track_temps)), 1) if track_temps else None
    umidade_media = round(float(np.mean(humidities)), 1) if humidities else None
    mean_wind_kmh = float(np.mean(winds_kmh)) if winds_kmh else None

    wet = 1 if any(r > 0 for r in rainfalls) else 0
    dry = 1 - wet
    warm = 1 if (temp_ar_media is not None and temp_ar_media >= warm_min) else 0
    cold = 1 if (temp_ar_media is not None and temp_ar_media <= cold_max) else 0
    windy = 1 if (mean_wind_kmh is not None and mean_wind_kmh >= windy_min) else 0
    cloudy = 0

    return {
        "wet": int(wet),
        "dry": int(dry),
        "warm": int(warm),
        "cold": int(cold),
        "windy": int(windy),
        "cloudy": int(cloudy),
        "temp_ar_media": temp_ar_media,
        "temp_pista_media": temp_pista_media,
        "umidade_media": umidade_media,
    }


def build_driver_lookup(
    race_drivers: List[Dict[str, Any]],
    quali_drivers: Optional[List[Dict[str, Any]]],
    constructors_map: Dict[str, str],
    warnings: Optional[List[str]] = None,
) -> Dict[int, Dict[str, str]]:
    """Builds per-race `driver_number -> {driver_key, constructor_key, full_name}` lookup."""
    lookup: Dict[int, Dict[str, str]] = {}

    def _ingest(records: List[Dict[str, Any]], overwrite: bool) -> None:
        for rec in records:
            num_raw = rec.get("driver_number")
            if not _is_number(num_raw):
                continue
            d_num = int(num_raw)
            if d_num in lookup and not overwrite:
                continue
            full_name = str(rec.get("full_name") or rec.get("broadcast_name") or "").strip()
            acronym = str(rec.get("name_acronym") or "").strip().upper()
            if not acronym:
                cleaned = re.sub(r"[^A-Za-z]", "", full_name).upper()
                acronym = cleaned[:3] if len(cleaned) >= 3 else f"D{d_num:02d}"
            else:
                acronym = acronym[:3]

            team_raw = rec.get("team_name")
            constructor_key = canonicalize_constructor(team_raw, constructors_map, warnings)
            lookup[d_num] = {
                "driver_key": acronym,
                "constructor_key": constructor_key,
                "full_name": full_name,
            }

    if quali_drivers:
        _ingest(quali_drivers, overwrite=True)
    _ingest(race_drivers or [], overwrite=True)
    return lookup


def resolve_starting_grid(
    race_starting_grid: List[Dict[str, Any]],
    quali_starting_grid: List[Dict[str, Any]],
    quali_results: List[Dict[str, Any]],
    quali_positions: List[Dict[str, Any]],
) -> Dict[int, int]:
    """Resolves official `posicao_grid` per `driver_number` with deterministic fallbacks."""
    grid: Dict[int, int] = {}

    def _apply(records: List[Dict[str, Any]]) -> None:
        for rec in records or []:
            d_num = rec.get("driver_number")
            pos = rec.get("position")
            if _is_number(d_num) and _is_number(pos) and int(pos) >= 1:
                d_int = int(d_num)
                if d_int not in grid:
                    grid[d_int] = int(pos)

    # 1. Primary: Race /starting_grid
    _apply(race_starting_grid)
    # 2. Fallback 1: Qualifying /starting_grid
    _apply(quali_starting_grid)
    # 3. Fallback 2a: Qualifying /session_result position
    _apply(quali_results)
    # 4. Fallback 2b: Latest chronological Qualifying /position
    if quali_positions:
        events: List[Tuple[datetime, int, int]] = []
        for rec in quali_positions:
            d_num = rec.get("driver_number")
            pos = rec.get("position")
            dt = parse_iso_utc(rec.get("date"))
            if _is_number(d_num) and _is_number(pos) and int(pos) >= 1 and dt is not None:
                events.append((dt, int(d_num), int(pos)))
        events.sort(key=lambda item: item[0])
        latest: Dict[int, int] = {}
        for _, d_int, pos_int in events:
            latest[d_int] = pos_int
        for d_int, pos_int in latest.items():
            if d_int not in grid:
                grid[d_int] = pos_int

    return grid


def compute_qualifying_times(
    quali_results: List[Dict[str, Any]],
    quali_laps: Optional[List[Dict[str, Any]]] = None,
) -> Dict[int, Tuple[Optional[float], Optional[float], Optional[float]]]:
    """Computes `(q_best, q_worst, q_avg)` per `driver_number` from Qualifying results/laps."""
    times_by_driver: Dict[int, List[float]] = {}

    for rec in quali_results or []:
        d_num = rec.get("driver_number")
        if not _is_number(d_num):
            continue
        d_int = int(d_num)
        dur = rec.get("duration")
        valid: List[float] = []
        if isinstance(dur, list):
            valid = [float(v) for v in dur if _is_number(v) and float(v) > 0]
        elif _is_number(dur) and float(dur) > 0:
            valid = [float(dur)]
        if valid:
            times_by_driver[d_int] = valid

    # Fallback from Qualifying /laps for drivers missing valid duration in /session_result
    if quali_laps:
        fallback_laps: Dict[int, List[float]] = {}
        for lap in quali_laps:
            d_num = lap.get("driver_number")
            dur = lap.get("lap_duration")
            is_out = lap.get("is_pit_out_lap")
            if _is_number(d_num) and _is_number(dur) and float(dur) > 0 and is_out is False:
                fallback_laps.setdefault(int(d_num), []).append(float(dur))
        for d_int, laps in fallback_laps.items():
            if d_int not in times_by_driver and laps:
                times_by_driver[d_int] = laps

    summary: Dict[int, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for d_int, vals in times_by_driver.items():
        if vals:
            summary[d_int] = (
                round(float(min(vals)), 3),
                round(float(max(vals)), 3),
                round(float(np.mean(vals)), 3),
            )
    return summary


def correlate_laps_and_positions(
    laps_records: List[Dict[str, Any]],
    position_records: List[Dict[str, Any]],
) -> Dict[Tuple[int, int], Optional[int]]:
    """Correlates Race `/laps` with `/position` via `lap_end_time` + forward-fill."""
    pos_times: Dict[int, List[datetime]] = {}
    pos_vals: Dict[int, List[int]] = {}

    grouped_pos: Dict[int, List[Tuple[datetime, int]]] = {}
    for rec in position_records or []:
        d_num = rec.get("driver_number")
        pos = rec.get("position")
        dt = parse_iso_utc(rec.get("date"))
        if _is_number(d_num) and _is_number(pos) and int(pos) >= 1 and dt is not None:
            grouped_pos.setdefault(int(d_num), []).append((dt, int(pos)))

    for d_int, evts in grouped_pos.items():
        evts.sort(key=lambda x: x[0])
        pos_times[d_int] = [e[0] for e in evts]
        pos_vals[d_int] = [e[1] for e in evts]

    laps_by_driver: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for lap in laps_records or []:
        d_num = lap.get("driver_number")
        l_num = lap.get("lap_number")
        if not (_is_number(d_num) and _is_number(l_num) and int(l_num) >= 1):
            continue
        d_int, l_int = int(d_num), int(l_num)
        existing = laps_by_driver.setdefault(d_int, {}).get(l_int)
        if existing is None or (
            not (_is_number(existing.get("lap_duration")) and float(existing["lap_duration"]) > 0)
            and _is_number(lap.get("lap_duration"))
            and float(lap["lap_duration"]) > 0
        ):
            laps_by_driver[d_int][l_int] = lap

    result: Dict[Tuple[int, int], Optional[int]] = {}
    for d_int, lap_map in laps_by_driver.items():
        sorted_lap_nums = sorted(lap_map.keys())
        times = pos_times.get(d_int, [])
        vals = pos_vals.get(d_int, [])
        last_known: Optional[int] = None

        for idx, l_num in enumerate(sorted_lap_nums):
            lap = lap_map[l_num]
            dt_start = parse_iso_utc(lap.get("date_start"))
            dur_raw = lap.get("lap_duration")
            dur = float(dur_raw) if _is_number(dur_raw) and float(dur_raw) > 0 else None

            lap_end: Optional[datetime] = None
            if dt_start is not None and dur is not None:
                lap_end = dt_start + timedelta(seconds=dur)
            else:
                # Commonly happens on Lap 1 in OpenF1 where lap_duration (and sometimes date_start) is null.
                if idx + 1 < len(sorted_lap_nums):
                    next_lap = lap_map[sorted_lap_nums[idx + 1]]
                    next_start = parse_iso_utc(next_lap.get("date_start"))
                    if next_start is not None:
                        lap_end = next_start
                if lap_end is None and dt_start is not None:
                    lap_end = dt_start

            assigned: Optional[int] = None
            if times and lap_end is not None:
                pos_idx = bisect_right(times, lap_end) - 1
                if pos_idx >= 0:
                    assigned = vals[pos_idx]
                    last_known = assigned
                elif dt_start is not None:
                    # Fallback if lap_end precedes the first position timestamp: pick first event after date_start
                    after_idx = bisect_right(times, dt_start)
                    if after_idx < len(vals):
                        assigned = vals[after_idx]
                        last_known = assigned
                    else:
                        assigned = vals[0]
                        last_known = assigned
            elif times and l_num == 1 and last_known is None:
                assigned = vals[0]
                last_known = assigned

            if assigned is None:
                assigned = last_known

            result[(d_int, l_num)] = assigned

    return result


def expand_stints(stints_records: List[Dict[str, Any]]) -> Dict[Tuple[int, int], Tuple[Optional[str], int]]:
    """Expands Race `/stints` into `(driver_number, lap_number) -> (compound, tyre_age)`."""
    sorted_stints = sorted(
        stints_records or [],
        key=lambda s: (
            int(s["driver_number"]) if _is_number(s.get("driver_number")) else -1,
            int(s["stint_number"]) if _is_number(s.get("stint_number")) else 0,
            int(s["lap_start"]) if _is_number(s.get("lap_start")) else 0,
        ),
    )

    lap_tyre_map: Dict[Tuple[int, int], Tuple[Optional[str], int]] = {}
    for st in sorted_stints:
        d_num = st.get("driver_number")
        lap_start = st.get("lap_start")
        lap_end = st.get("lap_end")
        if not (_is_number(d_num) and _is_number(lap_start) and _is_number(lap_end)):
            continue
        d_int = int(d_num)
        start_i = int(lap_start)
        end_i = int(lap_end)
        if start_i < 1 or end_i < start_i:
            continue

        raw_comp = str(st.get("compound") or "").strip().upper()
        compound: Optional[str] = raw_comp if raw_comp in _VALID_COMPOUNDS else None
        if compound is None:
            continue

        for lap in range(start_i, end_i + 1):
            lap_tyre_map[(d_int, lap)] = (compound, lap - start_i + 1)

    return lap_tyre_map


def aggregate_pits(
    pit_records: List[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int], float]]:
    """Aggregates Race `/pit` records per driver and per `(driver_number, lap_number)`."""
    driver_counts: Dict[int, int] = {}
    driver_durations: Dict[int, List[float]] = {}
    lap_pit_duration: Dict[Tuple[int, int], float] = {}

    seen_events: set[Tuple[int, int, str]] = set()

    for rec in pit_records or []:
        d_num = rec.get("driver_number")
        l_num = rec.get("lap_number")
        if not (_is_number(d_num) and _is_number(l_num) and int(l_num) >= 1):
            continue
        d_int, l_int = int(d_num), int(l_num)
        ev_key = (d_int, l_int, str(rec.get("date") or ""))
        if ev_key in seen_events:
            continue
        seen_events.add(ev_key)

        dur: Optional[float] = None
        for field in ("pit_duration", "stop_duration", "lane_duration"):
            val = rec.get(field)
            if _is_number(val) and float(val) > 0:
                dur = float(val)
                break

        driver_counts[d_int] = driver_counts.get(d_int, 0) + 1
        if dur is not None:
            driver_durations.setdefault(d_int, []).append(dur)
            # Keep same-lap pit duration (sum if multiple stops on the same lap)
            lap_pit_duration[(d_int, l_int)] = round(
                lap_pit_duration.get((d_int, l_int), 0.0) + dur, 3
            )
        else:
            lap_pit_duration.setdefault((d_int, l_int), 0.0)

    driver_agg: Dict[int, Dict[str, Any]] = {}
    for d_int, cnt in driver_counts.items():
        durs = driver_durations.get(d_int, [])
        if durs:
            total_dur = round(float(sum(durs)), 3)
            mean_dur: Optional[float] = round(float(np.mean(durs)), 6)
        else:
            total_dur = 0.0
            mean_dur = None
        driver_agg[d_int] = {
            "pit_count_acumulado": cnt,
            "pit_total_duration_seconds": total_dur,
            "pit_mean_duration_seconds": mean_dur,
        }

    return driver_agg, lap_pit_duration


def transform_race(
    meeting_info: Dict[str, Any],
    race_drivers: List[Dict[str, Any]],
    quali_drivers: List[Dict[str, Any]],
    race_results: List[Dict[str, Any]],
    quali_results: List[Dict[str, Any]],
    race_starting_grid: List[Dict[str, Any]],
    quali_starting_grid: List[Dict[str, Any]],
    quali_positions: List[Dict[str, Any]],
    quali_laps: List[Dict[str, Any]],
    race_weather: List[Dict[str, Any]],
    race_laps: List[Dict[str, Any]],
    race_positions: List[Dict[str, Any]],
    race_stints: List[Dict[str, Any]],
    race_pits: List[Dict[str, Any]],
    circuits_map: Dict[str, str],
    constructors_map: Dict[str, str],
    climate_thresholds: Dict[str, float],
    dob_lookup: Callable[[str, Optional[str]], Optional[date]],
    warnings: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Transforms all OpenF1 payloads for one GP into the 3 canonical DataFrames."""
    ano = int(meeting_info["year"])
    rodada = int(meeting_info["rodada"])
    nome_gp = str(meeting_info["meeting_name"]).strip()
    tipo_evento = str(meeting_info.get("tipo_evento") or "normal")
    race_date_str = str(meeting_info.get("date_start") or "")[:10]

    circuit_key = canonicalize_circuit(meeting_info.get("circuit_short_name"), circuits_map, warnings)
    race_key = f"{circuit_key}_{ano}_{rodada}"

    weather = aggregate_weather(race_weather, climate_thresholds)
    driver_map = build_driver_lookup(race_drivers, quali_drivers, constructors_map, warnings)
    grid_map = resolve_starting_grid(
        race_starting_grid, quali_starting_grid, quali_results, quali_positions
    )
    quali_times_map = compute_qualifying_times(quali_results, quali_laps)
    pit_agg_map, lap_pit_map = aggregate_pits(race_pits)
    lap_pos_map = correlate_laps_and_positions(race_laps, race_positions)
    lap_tyre_map = expand_stints(race_stints)

    # Deduplicate Race /laps by (driver_number, lap_number) & index fastest lap / max lap
    dedup_laps: Dict[Tuple[int, int], Dict[str, Any]] = {}
    fastest_lap_by_driver: Dict[int, float] = {}
    max_lap_by_driver: Dict[int, int] = {}

    for lap in race_laps or []:
        d_num = lap.get("driver_number")
        l_num = lap.get("lap_number")
        if not (_is_number(d_num) and _is_number(l_num) and int(l_num) >= 1):
            continue
        d_int, l_int = int(d_num), int(l_num)
        dur_raw = lap.get("lap_duration")
        dur = float(dur_raw) if _is_number(dur_raw) and float(dur_raw) > 0 else None

        existing = dedup_laps.get((d_int, l_int))
        if existing is None or (existing.get("_dur") is None and dur is not None):
            dedup_laps[(d_int, l_int)] = {"lap_number": l_int, "driver_number": d_int, "_dur": dur}

        if l_int > max_lap_by_driver.get(d_int, 0):
            max_lap_by_driver[d_int] = l_int
        if dur is not None:
            prev_fast = fastest_lap_by_driver.get(d_int)
            if prev_fast is None or dur < prev_fast:
                fastest_lap_by_driver[d_int] = dur

    # Filter Race /session_result:
    # - Exclude DNS (dns == True)
    # - Classified finishers keep their official integer `position >= 1`
    # - Started DNF / DSQ drivers whose OpenF1 `position` is None are ranked sequentially
    #   after classified finishers in FIA order (DNF before DSQ, descending laps completed,
    #   preserving OpenF1 order on ties), matching historical Kaggle `positionOrder` semantics.
    classified_entries: List[Tuple[int, int, Dict[str, Any]]] = []
    unclassified_starters: List[Tuple[int, int, int, Dict[str, Any]]] = []

    for orig_idx, res in enumerate(race_results or []):
        d_num = res.get("driver_number")
        if not _is_number(d_num):
            continue
        if res.get("dns") is True:
            continue

        pos = res.get("position")
        if _is_number(pos) and int(pos) >= 1:
            classified_entries.append((int(pos), orig_idx, res))
        elif res.get("dnf") is True or res.get("dsq") is True:
            d_int = int(d_num)
            laps_raw = res.get("number_of_laps")
            laps_val = int(laps_raw) if (_is_number(laps_raw) and int(laps_raw) >= 0) else max_lap_by_driver.get(d_int, 0)
            is_dsq_rank = 1 if res.get("dsq") is True else 0
            unclassified_starters.append((is_dsq_rank, -laps_val, orig_idx, res))

    classified_entries.sort(key=lambda item: (item[0], item[1]))
    unclassified_starters.sort(key=lambda item: (item[0], item[1], item[2]))

    next_pos = max((item[0] for item in classified_entries), default=0) + 1
    ordered_results: List[Tuple[int, Dict[str, Any]]] = [
        (pos_val, res) for pos_val, _, res in classified_entries
    ]
    for _, _, _, res in unclassified_starters:
        ordered_results.append((next_pos, res))
        next_pos += 1

    qualy_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    valid_driver_info: Dict[int, Dict[str, Any]] = {}
    seen_driver_keys: set[str] = set()

    for pos_final, res in ordered_results:
        d_int = int(res["driver_number"])
        info = driver_map.get(d_int)
        if info is None:
            msg = f"{race_key}: driver_number={d_int} missing from /drivers; skipped."
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)
            continue

        driver_key = info["driver_key"]
        if driver_key in seen_driver_keys:
            continue
        seen_driver_keys.add(driver_key)

        constructor_key = info["constructor_key"]
        dob = dob_lookup(driver_key, info.get("full_name"))
        idade = compute_driver_age(race_date_str, dob)

        posicao_grid = grid_map.get(d_int, 0)
        q_best, q_worst, q_avg = quali_times_map.get(d_int, (None, None, None))

        laps_raw = res.get("number_of_laps")
        if _is_number(laps_raw) and int(laps_raw) >= 0:
            voltas_completadas = int(laps_raw)
        else:
            voltas_completadas = max_lap_by_driver.get(d_int, 0)

        delta_pos = posicao_grid - pos_final
        is_overtake = 1 if (posicao_grid > 0 and pos_final < posicao_grid) else 0

        if pos_final == 1:
            diferenca_tempo: Optional[float] = 0.0
        else:
            gap_raw = res.get("gap_to_leader")
            if _is_number(gap_raw) and float(gap_raw) >= 0:
                diferenca_tempo = round(float(gap_raw), 3)
            else:
                diferenca_tempo = None

        fl_raw = fastest_lap_by_driver.get(d_int)
        fastest_lap_seconds = round(fl_raw, 3) if fl_raw is not None else None

        dsq_flag = res.get("dsq") is True
        dnf_flag = res.get("dnf") is True
        status_final = "DNF_other" if (dsq_flag or dnf_flag) else "Finished"

        pits = pit_agg_map.get(
            d_int,
            {
                "pit_count_acumulado": 0,
                "pit_total_duration_seconds": 0.0,
                "pit_mean_duration_seconds": None,
            },
        )

        qualy_rows.append(
            {
                "race_key": race_key,
                "driver_key": driver_key,
                "constructor_key": constructor_key,
                "circuit_key": circuit_key,
                "ano": ano,
                "rodada": rodada,
                "nome_gp": nome_gp,
                "tipo_evento": tipo_evento,
                "posicao_grid": posicao_grid,
                "q_best_seconds": q_best,
                "q_worst_seconds": q_worst,
                "q_avg_seconds": q_avg,
                "idade_piloto_corrida": idade,
                **weather,
            }
        )

        result_rows.append(
            {
                "race_key": race_key,
                "driver_key": driver_key,
                "constructor_key": constructor_key,
                "circuit_key": circuit_key,
                "ano": ano,
                "rodada": rodada,
                "nome_gp": nome_gp,
                "tipo_evento": tipo_evento,
                "posicao_grid": posicao_grid,
                "posicao_final": pos_final,
                "voltas_completadas": voltas_completadas,
                "delta_posicao_grid_final": delta_pos,
                "is_overtake": is_overtake,
                "diferenca_tempo_segundos": diferenca_tempo,
                "fastest_lap_seconds": fastest_lap_seconds,
                "idade_piloto_corrida": idade,
                **weather,
                "status_final": status_final,
                "pit_count_acumulado": pits["pit_count_acumulado"],
                "pit_total_duration_seconds": pits["pit_total_duration_seconds"],
                "pit_mean_duration_seconds": pits["pit_mean_duration_seconds"],
            }
        )

        valid_driver_info[d_int] = {
            "driver_key": driver_key,
            "constructor_key": constructor_key,
            "voltas_completadas": voltas_completadas,
        }

    # Build vw_voltas_timeseries strictly for drivers present in vw_resultados_com_clima
    ts_rows: List[Dict[str, Any]] = []
    for (d_int, l_int), lap in sorted(dedup_laps.items(), key=lambda x: (x[0][0], x[0][1])):
        drv = valid_driver_info.get(d_int)
        if drv is None:
            continue
        dur = lap["_dur"]
        tempo_volta = round(dur, 3) if dur is not None else None
        pos_volta = lap_pos_map.get((d_int, l_int))
        comp, tyre_age = lap_tyre_map.get((d_int, l_int), (None, None))
        is_pit = 1 if (d_int, l_int) in lap_pit_map else 0
        dur_pit = lap_pit_map.get((d_int, l_int), 0.0) if is_pit else 0.0

        ts_rows.append(
            {
                "race_key": race_key,
                "driver_key": drv["driver_key"],
                "constructor_key": drv["constructor_key"],
                "circuit_key": circuit_key,
                "ano": ano,
                "rodada": rodada,
                "tipo_evento": tipo_evento,
                "numero_volta": l_int,
                "tempo_volta_segundos": tempo_volta,
                "posicao_volta": pos_volta,
                "composto_atual": comp,
                "idade_pneu_volta": tyre_age,
                "is_pit_na_volta": is_pit,
                "duracao_pit_ultimas_1_volta": round(float(dur_pit), 3),
                **weather,
                "voltas_completadas_total": drv["voltas_completadas"],
            }
        )

    df_qualy = pd.DataFrame(qualy_rows, columns=QUALY_COLUMNS) if qualy_rows else pd.DataFrame(columns=QUALY_COLUMNS)
    df_results = pd.DataFrame(result_rows, columns=RESULTS_COLUMNS) if result_rows else pd.DataFrame(columns=RESULTS_COLUMNS)
    df_ts = pd.DataFrame(ts_rows, columns=TIMESERIES_COLUMNS) if ts_rows else pd.DataFrame(columns=TIMESERIES_COLUMNS)

    return df_qualy, df_results, df_ts
