"""Deterministic unit tests for OpenF1 pipeline transformers, client guards, DOB resolver, and CSV storage."""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.pipeline.dob_resolver import DobResolver
from src.pipeline.openf1_client import OpenF1Client, RateLimiter, is_session_completed
from src.pipeline.storage import (
    QUALY_FILENAME,
    RESULTS_FILENAME,
    TIMESERIES_FILENAME,
    atomic_write_view_csv,
    merge_view_data,
    read_view_csv,
)
from src.pipeline.transformers import (
    QUALY_COLUMNS,
    RESULTS_COLUMNS,
    TIMESERIES_COLUMNS,
    aggregate_pits,
    aggregate_weather,
    canonicalize_circuit,
    canonicalize_constructor,
    compute_driver_age,
    compute_qualifying_times,
    correlate_laps_and_positions,
    expand_stints,
    resolve_starting_grid,
    transform_race,
)

CONFIG_DIR = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "config"


@pytest.fixture(scope="module")
def configs():
    circuits = json.loads((CONFIG_DIR / "circuits_map.json").read_text(encoding="utf-8"))
    constructors = json.loads((CONFIG_DIR / "constructors_map.json").read_text(encoding="utf-8"))
    thresholds = json.loads((CONFIG_DIR / "climate_thresholds.json").read_text(encoding="utf-8"))
    return circuits, constructors, thresholds


def test_weather_flags_and_thresholds(configs):
    _, _, thresholds = configs

    # Warm + dry + windy (7.5 m/s = 27.0 km/h >= 25.0)
    warm_records = [
        {"air_temperature": 25.0, "track_temperature": 41.2, "humidity": 40.0, "wind_speed": 7.5, "rainfall": 0},
        {"air_temperature": 26.0, "track_temperature": 43.0, "humidity": 42.0, "wind_speed": 7.5, "rainfall": 0},
    ]
    w = aggregate_weather(warm_records, thresholds)
    assert (w["wet"], w["dry"], w["warm"], w["cold"], w["windy"], w["cloudy"]) == (0, 1, 1, 0, 1, 0)
    assert w["temp_ar_media"] == 25.5
    assert w["temp_pista_media"] == 42.1
    assert w["umidade_media"] == 41.0

    # Cold (<= 18.0) + wet
    cold_wet = [
        {"air_temperature": 15.7, "track_temperature": 18.9, "humidity": 78.4, "wind_speed": 3.0, "rainfall": 1},
    ]
    cw = aggregate_weather(cold_wet, thresholds)
    assert (cw["wet"], cw["dry"], cw["warm"], cw["cold"], cw["windy"], cw["cloudy"]) == (1, 0, 0, 1, 0, 0)

    # Empty telemetry fallback: dry=1, all flags ints, metrics None
    empty = aggregate_weather([], thresholds)
    assert (empty["wet"], empty["dry"], empty["warm"], empty["cold"], empty["windy"], empty["cloudy"]) == (0, 1, 0, 0, 0, 0)
    assert empty["temp_ar_media"] is None


def test_age_calculation_365_25():
    # Kimi Antonelli (2006-08-25) at Australian GP 2025-03-16 -> 6778 / 365.25 = 18.557 -> 18.56
    assert compute_driver_age("2025-03-16", date(2006, 8, 25)) == 18.56
    # Max Verstappen (1997-09-30) at Australian GP 2025-03-16 -> 27.46
    assert compute_driver_age("2025-03-16", "1997-09-30") == 27.46
    assert compute_driver_age("2025-03-16", None) is None


def test_circuit_and_constructor_canonicalization(configs):
    circuits, constructors, _ = configs
    warnings: list[str] = []

    assert canonicalize_circuit("Melbourne", circuits, warnings) == "albert_park"
    assert canonicalize_circuit("Spa-Francorchamps", circuits, warnings) == "spa"
    assert canonicalize_circuit("Montréal", circuits, warnings) == "villeneuve"
    assert canonicalize_circuit("Austin", circuits, warnings) == "americas"
    assert canonicalize_circuit("Yas Marina Circuit", circuits, warnings) == "yas_marina"
    assert canonicalize_circuit("New Circuit X", circuits, warnings) == "new_circuit_x"
    assert any("Unmapped circuit" in w for w in warnings)

    assert canonicalize_constructor("Kick Sauber", constructors, warnings) == "Audi"
    assert canonicalize_constructor("Stake F1 Team Kick Sauber", constructors, warnings) == "Audi"
    assert canonicalize_constructor("Audi", constructors, warnings) == "Audi"
    assert canonicalize_constructor("RB", constructors, warnings) == "Racing Bulls"
    assert canonicalize_constructor("Visa Cash App Racing Bulls", constructors, warnings) == "Racing Bulls"
    assert canonicalize_constructor("Red Bull Racing", constructors, warnings) == "Red Bull Racing"
    assert canonicalize_constructor("Cadillac", constructors, warnings) == "Cadillac"
    assert canonicalize_constructor("Alpine", constructors, warnings) == "Alpine"
    assert canonicalize_constructor("Haas F1 Team", constructors, warnings) == "Haas"


def test_grid_fallbacks_and_pitlane_zero():
    grid = resolve_starting_grid(
        race_starting_grid=[{"driver_number": 1, "position": 2}],
        quali_starting_grid=[{"driver_number": 4, "position": 1}],
        quali_results=[{"driver_number": 16, "position": 3}],
        quali_positions=[
            {"driver_number": 44, "position": 8, "date": "2025-03-15T05:10:00+00:00"},
            {"driver_number": 44, "position": 5, "date": "2025-03-15T05:55:00+00:00"},
        ],
    )
    assert grid == {1: 2, 4: 1, 16: 3, 44: 5}
    # Pitlane starter (driver 55 not in grid map) defaults to 0 inside transform_race


def test_qualifying_times_primary_and_laps_fallback():
    q_results = [
        {"driver_number": 1, "duration": [78.123, 77.500, 77.100]},
        {"driver_number": 4, "duration": [None, None, None]},
    ]
    q_laps = [
        # Driver 4 pit-out lap should be excluded
        {"driver_number": 4, "lap_duration": 105.0, "is_pit_out_lap": True},
        {"driver_number": 4, "lap_duration": 78.0, "is_pit_out_lap": False},
        {"driver_number": 4, "lap_duration": 77.4, "is_pit_out_lap": False},
    ]
    times = compute_qualifying_times(q_results, q_laps)
    assert times[1] == (77.1, 78.123, 77.574)
    assert times[4] == (77.4, 78.0, 77.7)


def test_lap_position_correlation_and_forward_fill():
    laps = [
        # Lap 1 has null lap_duration (common in OpenF1)
        {"driver_number": 1, "lap_number": 1, "date_start": "2025-03-16T04:00:00+00:00", "lap_duration": None},
        {"driver_number": 1, "lap_number": 2, "date_start": "2025-03-16T04:01:40+00:00", "lap_duration": 90.0},
        {"driver_number": 1, "lap_number": 3, "date_start": "2025-03-16T04:03:10+00:00", "lap_duration": 89.5},
    ]
    positions = [
        {"driver_number": 1, "position": 2, "date": "2025-03-16T04:00:05+00:00"},
        {"driver_number": 1, "position": 1, "date": "2025-03-16T04:02:50+00:00"},
    ]
    mapped = correlate_laps_and_positions(laps, positions)
    assert mapped[(1, 1)] == 2  # uses lap 2 date_start as lap 1 end
    assert mapped[(1, 2)] == 1  # overtaken at 04:02:50 <= 04:03:10
    assert mapped[(1, 3)] == 1  # forward-filled


def test_stint_expansion_and_pit_aggregation():
    stints = [
        {"driver_number": 1, "stint_number": 1, "compound": "medium", "lap_start": 1, "lap_end": 20},
        {"driver_number": 1, "stint_number": 2, "compound": "HARD", "lap_start": 21, "lap_end": 50},
    ]
    expanded = expand_stints(stints)
    assert expanded[(1, 1)] == ("MEDIUM", 1)
    assert expanded[(1, 20)] == ("MEDIUM", 20)
    assert expanded[(1, 21)] == ("HARD", 1)
    assert expanded[(1, 50)] == ("HARD", 30)

    pits = [
        {"driver_number": 1, "lap_number": 20, "pit_duration": 22.456, "date": "2025-03-16T04:30:00Z"},
        {"driver_number": 1, "lap_number": 40, "stop_duration": 23.100, "date": "2025-03-16T05:00:00Z"},
    ]
    drv_agg, lap_pits = aggregate_pits(pits)
    assert drv_agg[1]["pit_count_acumulado"] == 2
    assert drv_agg[1]["pit_total_duration_seconds"] == 45.556
    assert drv_agg[1]["pit_mean_duration_seconds"] == 22.778
    assert lap_pits[(1, 20)] == 22.456
    assert lap_pits[(1, 40)] == 23.1


def test_transform_race_end_to_end_and_idempotent_storage(configs, tmp_path):
    circuits, constructors, thresholds = configs
    dobs = {"NOR": date(1999, 11, 13), "VER": date(1997, 9, 30), "ANT": date(2006, 8, 25), "BOR": date(2004, 10, 14)}

    meeting = {
        "meeting_key": 1254,
        "year": 2025,
        "rodada": 1,
        "meeting_name": "Australian Grand Prix",
        "circuit_short_name": "Melbourne",
        "tipo_evento": "normal",
        "date_start": "2025-03-16T04:00:00+00:00",
    }
    drivers = [
        {"driver_number": 4, "name_acronym": "NOR", "full_name": "Lando NORRIS", "team_name": "McLaren"},
        {"driver_number": 1, "name_acronym": "VER", "full_name": "Max VERSTAPPEN", "team_name": "Red Bull Racing"},
        {"driver_number": 12, "name_acronym": "ANT", "full_name": "Kimi ANTONELLI", "team_name": "Mercedes"},
        {"driver_number": 5, "name_acronym": "BOR", "full_name": "Gabriel BORTOLETO", "team_name": "Kick Sauber"},
    ]
    race_results = [
        {"driver_number": 4, "position": 1, "number_of_laps": 57, "gap_to_leader": 0, "dnf": False, "dns": False, "dsq": False},
        {"driver_number": 1, "position": 2, "number_of_laps": 57, "gap_to_leader": 0.895, "dnf": False, "dns": False, "dsq": False},
        # Pitlane starter (absent from starting_grid), finished lapped ("+1 LAP")
        {"driver_number": 12, "position": 3, "number_of_laps": 56, "gap_to_leader": "+1 LAP", "dnf": False, "dns": False, "dsq": False},
        # DNF driver with OpenF1 position=None must still be classified after finishers
        {"driver_number": 5, "position": None, "number_of_laps": 12, "gap_to_leader": None, "dnf": True, "dns": False, "dsq": False},
        # DNS driver must be excluded
        {"driver_number": 6, "position": None, "number_of_laps": 0, "gap_to_leader": None, "dnf": False, "dns": True, "dsq": False},
    ]
    starting_grid = [
        {"driver_number": 4, "position": 1},
        {"driver_number": 1, "position": 3},
        {"driver_number": 5, "position": 15},
    ]
    quali_results = [
        {"driver_number": 4, "position": 1, "duration": [76.5, 76.0, 75.5]},
        {"driver_number": 1, "position": 3, "duration": [76.8, 76.2, 75.9]},
    ]
    weather = [
        {"air_temperature": 15.7, "track_temperature": 18.9, "humidity": 78.4, "wind_speed": 2.5, "rainfall": 1},
    ]
    laps = [
        {"driver_number": 4, "lap_number": 1, "date_start": "2025-03-16T04:00:00Z", "lap_duration": None},
        {"driver_number": 4, "lap_number": 2, "date_start": "2025-03-16T04:01:40Z", "lap_duration": 84.501},
        {"driver_number": 12, "lap_number": 1, "date_start": "2025-03-16T04:00:00Z", "lap_duration": 99.100},
    ]
    positions = [
        {"driver_number": 4, "position": 1, "date": "2025-03-16T04:00:10Z"},
        {"driver_number": 12, "position": 16, "date": "2025-03-16T04:00:10Z"},
    ]
    stints = [
        {"driver_number": 4, "stint_number": 1, "compound": "INTERMEDIATE", "lap_start": 1, "lap_end": 2},
    ]
    pits = [
        {"driver_number": 4, "lap_number": 2, "pit_duration": 21.5},
    ]

    df_q, df_r, df_t = transform_race(
        meeting_info=meeting,
        race_drivers=drivers,
        quali_drivers=[],
        race_results=race_results,
        quali_results=quali_results,
        race_starting_grid=starting_grid,
        quali_starting_grid=[],
        quali_positions=[],
        quali_laps=[],
        race_weather=weather,
        race_laps=laps,
        race_positions=positions,
        race_stints=stints,
        race_pits=pits,
        circuits_map=circuits,
        constructors_map=constructors,
        climate_thresholds=thresholds,
        dob_lookup=lambda dk, _: dobs.get(dk),
    )

    assert list(df_q.columns) == QUALY_COLUMNS
    assert list(df_r.columns) == RESULTS_COLUMNS
    assert list(df_t.columns) == TIMESERIES_COLUMNS
    assert len(df_q) == len(df_r) == 4  # DNS excluded

    r_by_drv = df_r.set_index("driver_key")
    # Winner gap == 0.0, pit aggregates populated
    assert r_by_drv.loc["NOR", "diferenca_tempo_segundos"] == 0.0
    assert r_by_drv.loc["NOR", "fastest_lap_seconds"] == 84.501
    assert r_by_drv.loc["NOR", "pit_count_acumulado"] == 1
    assert r_by_drv.loc["NOR", "pit_total_duration_seconds"] == 21.5
    # Overtake: VER started P3, finished P2 -> delta=+1, is_overtake=1
    assert r_by_drv.loc["VER", "delta_posicao_grid_final"] == 1
    assert r_by_drv.loc["VER", "is_overtake"] == 1
    # Pitlane starter ANT: posicao_grid=0 -> delta=-3, is_overtake=0, lapped -> NULL gap
    assert r_by_drv.loc["ANT", "posicao_grid"] == 0
    assert r_by_drv.loc["ANT", "delta_posicao_grid_final"] == -3
    assert r_by_drv.loc["ANT", "is_overtake"] == 0
    assert pd.isna(r_by_drv.loc["ANT", "diferenca_tempo_segundos"])
    # DNF driver BOR -> Kick Sauber canonicalized to Audi, DNF_other
    assert r_by_drv.loc["BOR", "constructor_key"] == "Audi"
    assert r_by_drv.loc["BOR", "status_final"] == "DNF_other"

    # Timeseries check: same-lap pit duration & stint expansion
    t_nor_2 = df_t[(df_t["driver_key"] == "NOR") & (df_t["numero_volta"] == 2)].iloc[0]
    assert t_nor_2["posicao_volta"] == 1
    assert t_nor_2["composto_atual"] == "INTERMEDIATE"
    assert t_nor_2["idade_pneu_volta"] == 2
    assert t_nor_2["is_pit_na_volta"] == 1
    assert t_nor_2["duracao_pit_ultimas_1_volta"] == 21.5

    # Storage atomic write + upsert idempotency (byte-for-byte identical)
    q_path = tmp_path / QUALY_FILENAME
    r_path = tmp_path / RESULTS_FILENAME
    t_path = tmp_path / TIMESERIES_FILENAME

    atomic_write_view_csv(df_q, q_path, QUALY_FILENAME)
    atomic_write_view_csv(df_r, r_path, RESULTS_FILENAME)
    atomic_write_view_csv(df_t, t_path, TIMESERIES_FILENAME)

    raw_r_first = r_path.read_bytes()
    assert not raw_r_first.startswith(b"\xef\xbb\xbf")  # no BOM
    assert b"\r\n" not in raw_r_first  # LF only
    assert b";NULL;" in raw_r_first

    # Re-upsert same round -> byte-for-byte identical
    merged_r = merge_view_data(
        read_view_csv(r_path, RESULTS_COLUMNS),
        df_r,
        RESULTS_FILENAME,
        replace_rounds=[(2025, 1)],
    )
    atomic_write_view_csv(merged_r, r_path, RESULTS_FILENAME)
    assert r_path.read_bytes() == raw_r_first


def test_is_session_completed_live_and_future_guard():
    now = datetime(2026, 9, 6, 16, 0, 0, tzinfo=timezone.utc)
    # Finished 3h ago -> completed
    assert is_session_completed({"date_end": "2026-09-06T13:00:00+00:00"}, now_utc=now) is True
    # Finished 1h ago (< 2h buffer) -> NOT completed
    assert is_session_completed({"date_end": "2026-09-06T15:00:00+00:00"}, now_utc=now) is False
    # Cancelled -> NOT completed
    assert is_session_completed({"date_end": "2026-09-06T10:00:00+00:00", "is_cancelled": True}, now_utc=now) is False
    # Missing date_end -> NOT completed
    assert is_session_completed({}, now_utc=now) is False


def test_dob_resolver_persists_newly_discovered_rookie(tmp_path):
    dob_file = tmp_path / "drivers_dob.json"
    dob_file.write_text('{"VER": "1997-09-30"}\n', encoding="utf-8")

    mock_session = MagicMock()
    search_resp = MagicMock()
    search_resp.json.return_value = {
        "search": [{"id": "Q123", "description": "British racing driver"}]
    }
    entity_resp = MagicMock()
    entity_resp.json.return_value = {
        "entities": {
            "Q123": {
                "descriptions": {"en": {"value": "British racing driver"}},
                "claims": {
                    "P569": [{"mainsnak": {"datavalue": {"value": {"time": "+2008-05-12T00:00:00Z"}}}}]
                },
            }
        }
    }
    mock_session.get.side_effect = [search_resp, entity_resp]

    resolver = DobResolver(dob_file=dob_file, enable_network_lookup=True, session=mock_session)
    assert resolver.get_dob("VER") == date(1997, 9, 30)
    assert resolver.get_dob("ROK", "Future Rookie") == date(2008, 5, 12)
    assert resolver.newly_discovered == {"ROK": "2008-05-12"}
    saved = json.loads(dob_file.read_text(encoding="utf-8"))
    assert saved["ROK"] == "2008-05-12"


def test_rate_limiter_enforces_second_and_minute_windows():
    clock = [100.0]
    sleeps: list[float] = []

    def fake_sleep(dt: float) -> None:
        sleeps.append(dt)
        clock[0] += dt

    limiter = RateLimiter(max_per_second=3, max_per_minute=28, time_func=lambda: clock[0], sleep_func=fake_sleep)
    for _ in range(4):
        limiter.acquire()
    assert len(sleeps) == 1
    assert sleeps[0] >= 1.0


def test_openf1_client_discover_excludes_testing_and_sprint():
    mock_session = MagicMock()
    meetings_resp = MagicMock(status_code=200)
    meetings_resp.json.return_value = [
        {"meeting_key": 100, "year": 2025, "meeting_name": "Pre-Season Testing", "date_start": "2025-02-26T07:00:00Z"},
        {"meeting_key": 101, "year": 2025, "meeting_name": "Chinese Grand Prix", "circuit_short_name": "Shanghai", "date_start": "2025-03-21T03:00:00Z"},
    ]
    sessions_resp = MagicMock(status_code=200)
    sessions_resp.json.return_value = [
        {"meeting_key": 100, "session_key": 9001, "session_name": "Day 1", "session_type": "Practice", "date_start": "2025-02-26T07:00:00Z"},
        {"meeting_key": 101, "session_key": 9101, "session_name": "Sprint Qualifying", "session_type": "Qualifying", "date_start": "2025-03-21T07:30:00Z"},
        {"meeting_key": 101, "session_key": 9102, "session_name": "Sprint", "session_type": "Race", "date_start": "2025-03-22T03:00:00Z"},
        {"meeting_key": 101, "session_key": 9103, "session_name": "Qualifying", "session_type": "Qualifying", "date_start": "2025-03-22T07:00:00Z"},
        {"meeting_key": 101, "session_key": 9104, "session_name": "Race", "session_type": "Race", "date_start": "2025-03-23T07:00:00Z", "date_end": "2025-03-23T09:00:00Z"},
    ]
    mock_session.get.side_effect = [meetings_resp, sessions_resp]

    client = OpenF1Client(session=mock_session, rate_limiter=RateLimiter(max_per_second=100, max_per_minute=100))
    gps = client.discover_gp_meetings(2025)
    assert len(gps) == 1
    assert gps[0]["rodada"] == 1
    assert gps[0]["tipo_evento"] == "sprint_weekend"
    assert gps[0]["race_session"]["session_key"] == 9104
    assert gps[0]["qualifying_session"]["session_key"] == 9103


def test_select_incremental_years_skips_completed_previous_season():
    from src.pipeline.sync import select_incremental_years

    # 2025 already has 24 rounds (>= 20) -> only inspect 2026
    synced_complete = {(2025, r) for r in range(1, 25)} | {(2026, r) for r in range(1, 10)}
    assert select_incremental_years(synced_complete, current_year=2026) == [2026]

    # 2025 incomplete (< 20 rounds) -> inspect both 2025 and 2026
    synced_incomplete = {(2025, r) for r in range(1, 15)}
    assert select_incremental_years(synced_incomplete, current_year=2026) == [2025, 2026]

