"""Small, read-only Handy settings and model-event probe."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

_MAX_SETTINGS_BYTES = 256 << 10
_MAX_LOG_BYTES = 512 << 10
_settings_cache: dict[str, tuple[tuple[int, int], str | None]] = {}
_log_cache: dict[tuple[str, float], tuple[tuple[int, int], tuple[str, str, str] | None]] = {}

_TIMESTAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2})\]\[(\d{2}:\d{2}:\d{2})\]")
_MODEL_EVENT = re.compile(
    r"(?:Loaded whisper model |Live preview: model |Live streaming transcription started \(model )'([^']+)'"
)
_UNLOAD_EVENT = re.compile(r"Model unloaded due to inactivity|Model idle for \d+s.*unloading", re.I)


@dataclass(frozen=True)
class HandyEvidence:
    selected_model: str | None = None
    last_model: str | None = None
    last_event: str | None = None
    last_event_at: str | None = None  # UTC ISO 8601, as Handy logs it

    @property
    def last_event_local(self) -> str:
        if not self.last_event_at:
            return ""
        try:
            return datetime.fromisoformat(self.last_event_at).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return self.last_event_at


def _signature(path: str) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def _selected_model(path: str) -> str | None:
    signature = _signature(path)
    if signature is None:
        return None
    hit = _settings_cache.get(path)
    if hit and hit[0] == signature:
        return hit[1]
    value = None
    if signature[1] <= _MAX_SETTINGS_BYTES:
        try:
            with open(path, "rb") as f:
                data = json.loads(f.read(_MAX_SETTINGS_BYTES + 1))
            selected = (data.get("settings") or {}).get("selected_model")
            value = selected if isinstance(selected, str) and selected else None
        except (OSError, ValueError, AttributeError):
            pass
    _settings_cache[path] = (signature, value)
    return value


def _last_model_event(path: str, process_started: float) -> tuple[str, str, str] | None:
    signature = _signature(path)
    if signature is None:
        return None
    key = (path, process_started)
    hit = _log_cache.get(key)
    if hit and hit[0] == signature:
        return hit[1]
    latest: tuple[str, str, str] | None = None
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - _MAX_LOG_BYTES))
            lines = f.read(_MAX_LOG_BYTES).decode("utf-8", "replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        match = _TIMESTAMP.match(line)
        if not match:
            continue
        try:
            occurred = datetime.strptime(
                f"{match[1]} {match[2]}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if process_started and occurred.timestamp() < process_started - 2:
            continue
        event = None
        model_match = _MODEL_EVENT.search(line)
        if model_match:
            if "Loaded whisper model " in line:
                event = "loaded"
            elif "Live streaming transcription started " in line:
                event = "streaming started"
            else:
                event = "preview"
            latest = (event, model_match[1], occurred.isoformat())
        elif _UNLOAD_EVENT.search(line):
            event = "unloaded" if "unloaded" in line.lower() else "unloading"
            model = latest[1] if latest else ""
            latest = (event, model, occurred.isoformat())
    _log_cache[key] = (signature, latest)
    return latest


def inspect_handy(
    process_started: float,
    appdata: str | None = None,
    localappdata: str | None = None,
) -> HandyEvidence | None:
    """Read Handy's selected model and latest model event after this process started."""
    if sys.platform == "win32":
        default_data = os.environ.get("APPDATA", "")
        default_local = os.environ.get("LOCALAPPDATA", "")
    else:  # Tauri keeps both the settings store and logs/ under the XDG data dir on Linux
        default_data = default_local = os.environ.get("XDG_DATA_HOME") or os.path.join(
            os.path.expanduser("~"), ".local", "share")
    appdata = default_data if appdata is None else appdata
    localappdata = default_local if localappdata is None else localappdata
    settings_path = os.path.join(appdata, "com.pais.handy", "settings_store.json")
    log_path = os.path.join(localappdata, "com.pais.handy", "logs", "handy.log")
    selected = _selected_model(settings_path)
    event = _last_model_event(log_path, process_started)
    if not selected and not event:
        return None
    return HandyEvidence(
        selected_model=selected,
        last_model=event[1] or None if event else None,
        last_event=event[0] if event else None,
        last_event_at=event[2] if event else None,
    )
