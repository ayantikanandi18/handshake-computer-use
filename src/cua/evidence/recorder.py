"""
Evidence: what happened, why, and enough to debug it after the fact.

Two rules shape this module.

First, everything is redacted on the way in, not on the way out. A log that holds a
member's name until some later scrubbing step is a log that has already leaked it into
a backup. Redaction at the write boundary is the only version that actually holds.

Second, failures get a richer signal than successes. A structured event stream is enough
to understand a run that worked; a run that broke needs the screen. Screenshots on every
step would be wasteful, so the recorder keeps the last screen and promotes it on failure.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Patterns that must never reach disk in the clear. Deliberately broad: over-redaction
# costs a debugging round trip, under-redaction is a reportable incident.
REDACTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED:SSN]"),
    (re.compile(r"\b\d{13,19}\b"), "[REDACTED:PAN]"),
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key)\b\s*[:=]\s*\S+"),
     r"\1=[REDACTED]"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[REDACTED:EMAIL]"),
]

SENSITIVE_KEYS = {"password", "passwd", "pwd", "secret", "token", "api_key", "apikey",
                  "ssn", "tax_id", "card", "pan", "authorization"}


def redact_text(value: str) -> str:
    out = value
    for pattern, replacement in REDACTIONS:
        out = pattern.sub(replacement, out)
    return out


def redact(value: Any) -> Any:
    """Recursively scrub a structure before it is persisted."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = redact(item)
        return cleaned
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


class EvidenceRecorder:
    """Writes a run's evidence into its own directory."""

    def __init__(self, root: str | Path, run_id: str | None = None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.dir = Path(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.dir / "events.jsonl"
        self._started = time.time()

    def path(self, filename: str) -> str:
        return str(self.dir / filename)

    def log(self, event: str, data: dict[str, Any] | None = None) -> None:
        record = {
            "t": round(time.time() - self._started, 3),
            "at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "data": redact(data or {}),
        }
        with self._events_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(record) + "\n")

    def write_json(self, filename: str, payload: Any) -> str:
        target = self.dir / filename
        target.write_text(
            json.dumps(redact(payload), indent=2, default=str), encoding="utf-8", newline=""
        )
        return str(target)

    def write_text(self, filename: str, text: str) -> str:
        target = self.dir / filename
        target.write_text(redact_text(text), encoding="utf-8", newline="")
        return str(target)

    def capture_failure(self, surface, step_id: str, expected: str, observed: str) -> dict:
        """The richer signal a failure earns: screen, text, and what we thought we'd see."""
        detail: dict[str, Any] = {
            "step": step_id, "expected": expected, "observed": observed[:800],
        }
        try:
            observation = surface.observe(
                screenshot=True, screenshot_path=self.path(f"FAIL_{step_id}.png")
            )
            detail["screenshot"] = self.path(f"FAIL_{step_id}.png")
            detail["location"] = observation.location
            self.write_text(f"FAIL_{step_id}.txt", observation.summarise(limit=200))
            detail["screen_dump"] = self.path(f"FAIL_{step_id}.txt")
        except Exception as exc:  # evidence capture must never mask the original failure
            detail["capture_error"] = str(exc)[:200]
        self.log("failure", detail)
        return detail
