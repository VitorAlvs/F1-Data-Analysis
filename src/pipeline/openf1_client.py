"""Resilient OpenF1 API client with dual sliding-window rate limiting and optional disk cache."""

from __future__ import annotations

import collections
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Deque, Dict, List, Optional

import requests

OPENF1_BASE_URL = "https://api.openf1.org/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 5
DEFAULT_BACKOFF_BASE = 2.0
DEFAULT_USER_AGENT = (
    "F1-Data-Analysis-Pipeline/1.0 (+https://github.com/VitorAlvs/F1-Data-Analysis)"
)

ALLOWED_ENDPOINTS = frozenset(
    {
        "meetings",
        "sessions",
        "drivers",
        "session_result",
        "starting_grid",
        "weather",
        "position",
        "laps",
        "pit",
        "stints",
    }
)


def parse_iso_utc(date_str: Optional[str]) -> Optional[datetime]:
    """Parses an ISO-8601 timestamp string into a timezone-aware UTC datetime."""
    if not date_str or not isinstance(date_str, str):
        return None
    cleaned = date_str.strip()
    if not cleaned:
        return None
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_session_completed(
    race_session: Dict[str, Any],
    now_utc: Optional[datetime] = None,
    buffer_hours: float = 2.0,
) -> bool:
    """Returns True only if the Race session has ended at least `buffer_hours` ago."""
    if race_session.get("is_cancelled") is True:
        return False
    end_dt = parse_iso_utc(race_session.get("date_end"))
    if end_dt is None:
        return False
    ref_now = now_utc or datetime.now(timezone.utc)
    if ref_now.tzinfo is None:
        ref_now = ref_now.replace(tzinfo=timezone.utc)
    else:
        ref_now = ref_now.astimezone(timezone.utc)
    return end_dt + timedelta(hours=buffer_hours) <= ref_now


class RateLimiter:
    """Thread-safe dual sliding-window rate limiter (per-second and per-minute)."""

    def __init__(
        self,
        max_per_second: int = 3,
        max_per_minute: int = 28,
        time_func: Optional[Callable[[], float]] = None,
        sleep_func: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.max_per_second = max_per_second
        self.max_per_minute = max_per_minute
        self.time_func = time_func or time.monotonic
        self.sleep_func = sleep_func or time.sleep
        self._second_window: Deque[float] = collections.deque()
        self._minute_window: Deque[float] = collections.deque()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Blocks until a request slot is available in both rolling windows."""
        with self._lock:
            while True:
                now = self.time_func()
                while self._second_window and self._second_window[0] <= now - 1.0:
                    self._second_window.popleft()
                while self._minute_window and self._minute_window[0] <= now - 60.0:
                    self._minute_window.popleft()

                wait_sec = 0.0
                if len(self._second_window) >= self.max_per_second:
                    wait_sec = self._second_window[0] + 1.0 - now

                wait_min = 0.0
                if len(self._minute_window) >= self.max_per_minute:
                    wait_min = self._minute_window[0] + 60.0 - now

                wait = max(wait_sec, wait_min)
                if wait > 0.0:
                    self.sleep_func(wait + 0.01)
                else:
                    self._second_window.append(now)
                    self._minute_window.append(now)
                    return now


class OpenF1Client:
    """HTTP client for OpenF1 v1 API with proactive throttling, retries, and optional cache."""

    def __init__(
        self,
        base_url: str = OPENF1_BASE_URL,
        cache_dir: Optional[Path] = None,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        rate_limiter: Optional[RateLimiter] = None,
        session: Optional[requests.Session] = None,
        sleep_func: Optional[Callable[[float], None]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.rate_limiter = rate_limiter or RateLimiter()
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", DEFAULT_USER_AGENT)
        self.sleep_func = sleep_func or time.sleep
        self.logger = logger or logging.getLogger(__name__)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, endpoint: str, params: Dict[str, Any]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        if "session_key" in params:
            folder = self.cache_dir / f"session_{params['session_key']}"
            rest = {k: v for k, v in sorted(params.items()) if k != "session_key"}
        elif "meeting_key" in params:
            folder = self.cache_dir / f"meeting_{params['meeting_key']}"
            rest = {k: v for k, v in sorted(params.items()) if k != "meeting_key"}
        elif "year" in params:
            folder = self.cache_dir / f"year_{params['year']}"
            rest = {k: v for k, v in sorted(params.items()) if k != "year"}
        else:
            folder = self.cache_dir / "common"
            rest = dict(sorted(params.items()))

        if not rest:
            filename = f"{endpoint}.json"
        else:
            suffix = "_".join(f"{k}-{v}" for k, v in rest.items())
            suffix = re.sub(r"[^\w\-.]", "_", suffix)
            filename = f"{endpoint}__{suffix}.json"
        return folder / filename

    def _read_cache(self, path: Optional[Path]) -> Optional[List[Dict[str, Any]]]:
        if path is None or not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except Exception as exc:
            self.logger.warning("Corrupt cache file %s ignored: %s", path, exc)
        return None

    def _write_cache(self, path: Optional[Path], data: List[Dict[str, Any]]) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
        except Exception as exc:
            self.logger.warning("Failed to write cache file %s: %s", path, exc)

    def _get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        if endpoint not in ALLOWED_ENDPOINTS:
            raise ValueError(
                f"Endpoint '/{endpoint}' is not allowed. Supported: {sorted(ALLOWED_ENDPOINTS)}"
            )

        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        cpath = self._cache_path(endpoint, clean_params)

        if not force_refresh:
            cached = self._read_cache(cpath)
            if cached is not None:
                return cached

        url = f"{self.base_url}/{endpoint}"
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries):
            self.rate_limiter.acquire()
            try:
                resp = self.session.get(url, params=clean_params, timeout=self.timeout)

                if resp.status_code == 200:
                    payload = resp.json()
                    if isinstance(payload, list):
                        self._write_cache(cpath, payload)
                        return payload
                    self.logger.warning("Unexpected non-list JSON from %s: %s", url, type(payload))
                    return []

                if resp.status_code == 404:
                    self._write_cache(cpath, [])
                    return []

                if resp.status_code == 429:
                    backoff = self.backoff_base ** attempt
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            backoff = max(backoff, float(retry_after))
                        except ValueError:
                            pass
                    self.logger.warning(
                        "HTTP 429 on %s %s; retrying in %.2fs (attempt %d/%d)",
                        url,
                        clean_params,
                        backoff,
                        attempt + 1,
                        self.max_retries,
                    )
                    self.sleep_func(backoff)
                    continue

                if 500 <= resp.status_code < 600:
                    backoff = self.backoff_base ** attempt
                    self.logger.warning(
                        "HTTP %d on %s %s; retrying in %.2fs (attempt %d/%d)",
                        resp.status_code,
                        url,
                        clean_params,
                        backoff,
                        attempt + 1,
                        self.max_retries,
                    )
                    self.sleep_func(backoff)
                    continue

                resp.raise_for_status()

            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                backoff = self.backoff_base ** attempt
                self.logger.warning(
                    "Network error (%s) on %s %s; retrying in %.2fs (attempt %d/%d)",
                    type(exc).__name__,
                    url,
                    clean_params,
                    backoff,
                    attempt + 1,
                    self.max_retries,
                )
                self.sleep_func(backoff)

        if last_exc is not None:
            raise last_exc
        raise requests.HTTPError(f"Failed GET {url} ({clean_params}) after {self.max_retries} attempts.")

    # ── Endpoint Wrappers ─────────────────────────────────────────────────────

    def get_meetings(self, year: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self._get("meetings", params={"year": year}, force_refresh=force_refresh)

    def get_sessions(
        self,
        year: Optional[int] = None,
        meeting_key: Optional[int] = None,
        session_name: Optional[str] = None,
        session_type: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        return self._get(
            "sessions",
            params={
                "year": year,
                "meeting_key": meeting_key,
                "session_name": session_name,
                "session_type": session_type,
            },
            force_refresh=force_refresh,
        )

    def get_drivers(self, session_key: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self._get("drivers", params={"session_key": session_key}, force_refresh=force_refresh)

    def get_session_result(
        self, session_key: int, force_refresh: bool = False
    ) -> List[Dict[str, Any]]:
        return self._get(
            "session_result", params={"session_key": session_key}, force_refresh=force_refresh
        )

    def get_starting_grid(
        self,
        session_key: Optional[int] = None,
        meeting_key: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        return self._get(
            "starting_grid",
            params={"session_key": session_key, "meeting_key": meeting_key},
            force_refresh=force_refresh,
        )

    def get_weather(self, session_key: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self._get("weather", params={"session_key": session_key}, force_refresh=force_refresh)

    def get_position(self, session_key: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self._get(
            "position", params={"session_key": session_key}, force_refresh=force_refresh
        )

    def get_laps(self, session_key: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self._get("laps", params={"session_key": session_key}, force_refresh=force_refresh)

    def get_pit(self, session_key: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self._get("pit", params={"session_key": session_key}, force_refresh=force_refresh)

    def get_stints(self, session_key: int, force_refresh: bool = False) -> List[Dict[str, Any]]:
        return self._get("stints", params={"session_key": session_key}, force_refresh=force_refresh)

    # ── Calendar Discovery ────────────────────────────────────────────────────

    def discover_gp_meetings(
        self,
        year: int,
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Discovers all non-cancelled, non-testing Grand Prix meetings for `year`.

        Orders valid GP meetings chronologically by Race `date_start` and assigns
        sequential championship round numbers (`rodada` 1..N).
        """
        raw_meetings = self.get_meetings(year=year, force_refresh=force_refresh)
        raw_sessions = self.get_sessions(year=year, force_refresh=force_refresh)

        sessions_by_meeting: Dict[int, List[Dict[str, Any]]] = {}
        for sess in raw_sessions:
            m_key = sess.get("meeting_key")
            if m_key is not None:
                sessions_by_meeting.setdefault(int(m_key), []).append(sess)

        seen_keys: set[int] = set()
        gp_meetings: List[Dict[str, Any]] = []

        for meeting in raw_meetings:
            m_key_raw = meeting.get("meeting_key")
            if m_key_raw is None:
                continue
            m_key = int(m_key_raw)
            if m_key in seen_keys:
                continue
            seen_keys.add(m_key)

            if meeting.get("is_cancelled") is True:
                continue
            if meeting.get("year") is not None and int(meeting["year"]) != year:
                continue

            m_name = str(meeting.get("meeting_name") or "").strip()
            m_off = str(meeting.get("meeting_official_name") or "").strip()
            name_ctx = f"{m_name} {m_off}".lower()
            if "testing" in name_ctx or "pre-season test" in name_ctx:
                continue

            sessions = [
                s for s in sessions_by_meeting.get(m_key, []) if s.get("is_cancelled") is not True
            ]
            race_candidates: List[Dict[str, Any]] = []
            quali_candidates: List[Dict[str, Any]] = []
            has_sprint = False

            for s in sessions:
                s_name = str(s.get("session_name") or "").strip().lower()
                s_type = str(s.get("session_type") or "").strip().lower()

                if "sprint" in s_name or "sprint" in s_type or "shootout" in s_name:
                    has_sprint = True
                elif s_name == "race" or (
                    s_type == "race"
                    and "qualifying" not in s_name
                    and "practice" not in s_name
                ):
                    race_candidates.append(s)
                elif s_name == "qualifying" or (
                    s_type == "qualifying" and "sprint" not in s_name and "shootout" not in s_name
                ):
                    quali_candidates.append(s)

            if not race_candidates:
                continue

            race_candidates.sort(
                key=lambda s: (str(s.get("date_start") or ""), int(s.get("session_key") or 0)),
                reverse=True,
            )
            quali_candidates.sort(
                key=lambda s: (str(s.get("date_start") or ""), int(s.get("session_key") or 0)),
                reverse=True,
            )

            primary_race = race_candidates[0]
            primary_quali = quali_candidates[0] if quali_candidates else None

            gp_meetings.append(
                {
                    "meeting_key": m_key,
                    "year": year,
                    "meeting_name": m_name or f"Grand Prix {m_key}",
                    "circuit_short_name": (
                        meeting.get("circuit_short_name")
                        or primary_race.get("circuit_short_name")
                        or "unknown"
                    ),
                    "race_session": primary_race,
                    "qualifying_session": primary_quali,
                    "tipo_evento": "sprint_weekend" if has_sprint else "normal",
                    "date_start": primary_race.get("date_start") or meeting.get("date_start") or "",
                }
            )

        gp_meetings.sort(key=lambda item: (str(item["date_start"]), int(item["meeting_key"])))
        for idx, item in enumerate(gp_meetings, start=1):
            item["rodada"] = idx

        return gp_meetings
