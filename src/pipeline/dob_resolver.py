"""Driver Date-of-Birth resolver using local JSON seed + lightweight Wikimedia REST fallback."""

from __future__ import annotations

from datetime import date
import json
import logging
import os
from pathlib import Path
import re
from typing import Dict, Optional, Set

import requests

DEFAULT_DOB_PATH = Path(__file__).resolve().parent / "config" / "drivers_dob.json"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "F1DataAnalysisBot/1.0 (https://github.com/VitorAlvs/F1-Data-Analysis; f1-data-analysis@users.noreply.github.com)"

_MOTORSPORT_KEYWORDS = (
    "racing driver",
    "formula one",
    "formula 1",
    "motorsport",
    "f1 driver",
    "auto racing",
    "grand prix",
)


def _validate_iso_dob(dob_str: Optional[str]) -> Optional[date]:
    """Parses and validates a driver DOB within a plausible F1 birth-year range."""
    if not dob_str or not re.match(r"^\d{4}-\d{2}-\d{2}$", dob_str):
        return None
    try:
        parsed = date.fromisoformat(dob_str)
    except ValueError:
        return None
    if not (1940 <= parsed.year <= 2015):
        return None
    return parsed


class DobResolver:
    """Resolves driver DOBs from `drivers_dob.json` with optional Wikimedia REST fallback."""

    def __init__(
        self,
        dob_file: Path = DEFAULT_DOB_PATH,
        enable_network_lookup: bool = True,
        persist_updates: bool = True,
        session: Optional[requests.Session] = None,
        timeout: int = 12,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.dob_file = Path(dob_file)
        self.enable_network_lookup = enable_network_lookup
        self.persist_updates = persist_updates
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.session.headers.setdefault("Api-User-Agent", USER_AGENT)
        self.timeout = timeout
        self.logger = logger or logging.getLogger(__name__)

        self._dob_map: Dict[str, str] = self._load()
        self._lookup_attempted: Set[str] = set()
        self.newly_discovered: Dict[str, str] = {}
        self.missing_dobs: Set[str] = set()

    def _load(self) -> Dict[str, str]:
        if not self.dob_file.is_file():
            return {}
        raw = json.loads(self.dob_file.read_text(encoding="utf-8"))
        return {str(k).strip().upper(): str(v).strip() for k, v in raw.items() if v}

    def _save(self) -> None:
        if not self.persist_updates:
            return
        self.dob_file.parent.mkdir(parents=True, exist_ok=True)
        ordered = dict(sorted(self._dob_map.items()))
        payload = json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"
        tmp = self.dob_file.with_suffix(self.dob_file.suffix + f".tmp.{os.getpid()}")
        try:
            tmp.write_text(payload, encoding="utf-8", newline="\n")
            os.replace(tmp, self.dob_file)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _lookup_wikidata(self, full_name: str) -> Optional[str]:
        """Queries Wikidata API for claim P569 (date of birth) on a racing driver entity."""
        try:
            search_resp = self.session.get(
                WIKIDATA_API_URL,
                params={
                    "action": "wbsearchentities",
                    "search": full_name,
                    "language": "en",
                    "type": "item",
                    "limit": 5,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            search_resp.raise_for_status()
            candidates = search_resp.json().get("search", [])
        except Exception as exc:
            self.logger.warning("Wikidata search failed for '%s': %s", full_name, exc)
            return None

        candidate_ids = []
        for cand in candidates:
            desc = str(cand.get("description") or "").lower()
            if any(kw in desc for kw in _MOTORSPORT_KEYWORDS):
                candidate_ids.insert(0, cand["id"])
            elif cand.get("id"):
                candidate_ids.append(cand["id"])

        for qid in candidate_ids[:3]:
            try:
                ent_resp = self.session.get(
                    WIKIDATA_API_URL,
                    params={
                        "action": "wbgetentities",
                        "ids": qid,
                        "props": "claims|descriptions",
                        "languages": "en",
                        "format": "json",
                    },
                    timeout=self.timeout,
                )
                ent_resp.raise_for_status()
                entity = ent_resp.json().get("entities", {}).get(qid, {})
                desc = str(entity.get("descriptions", {}).get("en", {}).get("value") or "").lower()
                if desc and not any(kw in desc for kw in _MOTORSPORT_KEYWORDS):
                    continue
                claims = entity.get("claims", {}).get("P569", [])
                for claim in claims:
                    time_val = (
                        claim.get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                        .get("time", "")
                    )
                    match = re.match(r"^[+-]?(\d{4}-\d{2}-\d{2})T", str(time_val))
                    if match and _validate_iso_dob(match.group(1)):
                        return match.group(1)
            except Exception as exc:
                self.logger.warning("Wikidata entity fetch failed for %s: %s", qid, exc)
        return None

    def _lookup_wikipedia_wikitext(self, full_name: str) -> Optional[str]:
        """Fallback: extracts `{{birth date...|YYYY|MM|DD}}` from Wikipedia MediaWiki JSON API."""
        try:
            resp = self.session.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "titles": full_name.title(),
                    "redirects": 1,
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                    "rvsection": 0,
                    "format": "json",
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                revs = page.get("revisions") or []
                if not revs:
                    continue
                content = revs[0].get("slots", {}).get("main", {}).get("*", "")
                if not any(kw in content.lower() for kw in _MOTORSPORT_KEYWORDS):
                    continue
                m = re.search(
                    r"\{\{\s*birth\s+date(?:\s+and\s+age)?[^}]*\|(\d{4})\|(\d{1,2})\|(\d{1,2})",
                    content,
                    flags=re.IGNORECASE,
                )
                if m:
                    candidate = f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
                    if _validate_iso_dob(candidate):
                        return candidate
        except Exception as exc:
            self.logger.warning("Wikipedia API lookup failed for '%s': %s", full_name, exc)
        return None

    def get_dob(self, driver_key: str, full_name: Optional[str] = None) -> Optional[date]:
        """Returns driver's `datetime.date` DOB or `None` if unavailable."""
        key = str(driver_key).strip().upper()
        if not key:
            return None

        if key in self._dob_map:
            return _validate_iso_dob(self._dob_map[key])

        if key in self._lookup_attempted:
            return None
        self._lookup_attempted.add(key)

        if self.enable_network_lookup and full_name and str(full_name).strip():
            clean_name = " ".join(str(full_name).strip().split())
            dob_str = self._lookup_wikidata(clean_name) or self._lookup_wikipedia_wikitext(
                clean_name
            )
            parsed = _validate_iso_dob(dob_str)
            if parsed is not None and dob_str is not None:
                self._dob_map[key] = dob_str
                self.newly_discovered[key] = dob_str
                self._save()
                self.logger.info("Discovered DOB for new driver %s (%s): %s", key, clean_name, dob_str)
                return parsed

        self.missing_dobs.add(key)
        self.logger.warning(
            "DOB missing for driver_key=%s (full_name=%s); idade_piloto_corrida will be NULL.",
            key,
            full_name,
        )
        return None
