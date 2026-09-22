"""
Runs the same recorded capability against every runtime condition it has to survive.

This is the evidence for requirement 3.3. A capability that only works on the happy
path is not useful in production, so the interesting question is not "does replay work"
but "does replay tell the caller the truth when the app does something else".

Each scenario below is a condition that legitimately occurs in a back-office app, and
each one asserts which *class* of result the caller should get back - not just that
something happened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cua.artifact.schema import Capability
from cua.cli import DEMO_SECRETS, TARGET
from cua.escalation.broker import EscalationBroker
from cua.evidence.recorder import EvidenceRecorder
from cua.replay.engine import ReplayEngine
from cua.replay.outcomes import ResultStatus
from cua.safety.policy import PolicyEngine, default_policy_for_demo
from cua.surface.web import open_web_surface

REPO = Path(__file__).resolve().parents[1]
CAPABILITY = REPO / "capabilities" / "meridian.member.read_savings_balance.v1.json"


def set_faults(**flags) -> None:
    httpx.post(f"{TARGET}/admin/faults/api", json=flags, timeout=10)


def reset_faults() -> None:
    set_faults(session_timeout=False, permission_denied=False, surprise_dialog=False,
               slow_load=False, app_error=False)


SCENARIOS = [
    # name,                param member,  faults,                        expected status
    ("happy path",              "12345", {},                              ResultStatus.SUCCESS),
    ("different member",        "22841", {},                              ResultStatus.SUCCESS),
    ("member does not exist",   "99999", {},                              ResultStatus.BUSINESS_OUTCOME),
    ("restricted member",       "44190", {},                              ResultStatus.BUSINESS_OUTCOME),
    ("surprise interstitial",   "12345", {"surprise_dialog": True},       ResultStatus.SUCCESS),
    ("transient slow load",     "12345", {"slow_load": True},             ResultStatus.SUCCESS),
    ("session expires mid-flow","12345", {"session_timeout": True},       ResultStatus.SUCCESS),
    ("application 500",         "12345", {"app_error": True},             ResultStatus.FAILURE),
]


def main() -> int:
    capability = Capability.model_validate_json(CAPABILITY.read_text(encoding="utf-8"))
    rows = []
    failures = 0

    for name, member, faults, expected in SCENARIOS:
        reset_faults()
        if faults:
            set_faults(**faults)

        evidence = EvidenceRecorder(REPO / "evidence", run_id=f"matrix-{name.replace(' ', '-')}")
        surface = open_web_surface(headless=True)
        broker = EscalationBroker(evidence)
        try:
            engine = ReplayEngine(
                surface,
                PolicyEngine(default_policy_for_demo(TARGET), capability_id=capability.id),
                evidence,
                escalation=broker,
            )
            result = engine.run(capability, params={"member_id": member}, secrets=DEMO_SECRETS)
        finally:
            surface.close()
            reset_faults()

        ok = result.status is expected
        failures += 0 if ok else 1
        detail = result.outcome_code or (
            result.failure_kind.value if result.failure_kind else json.dumps(result.outputs)
        )
        recoveries = ",".join(
            f"{r.condition_code}:{'ok' if r.succeeded else 'failed'}" for r in result.recoveries
        )
        rows.append((name, member, result.status.value, detail, recoveries, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:26} -> {result.status.value:17} {str(detail)[:34]:36} {recoveries}")

    print()
    print(f"{len(rows) - failures}/{len(rows)} scenarios behaved as specified")
    (REPO / "evidence" / "matrix_summary.json").write_text(
        json.dumps(
            [
                {"scenario": r[0], "member_id": r[1], "status": r[2], "detail": r[3],
                 "recoveries": r[4], "as_expected": r[5]}
                for r in rows
            ],
            indent=2,
        ),
        encoding="utf-8",
        newline="",
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
