"""Command line entry points: discover, replay, operator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cua.artifact.schema import (
    BusinessOutcome, Capability, LocatorStrategy, LocatorTier, Output, Param,
    RecoverableCondition, RecoveryAction, Sensitivity, StateAssertion, SurfaceKind, Target,
)
from cua.discovery.agent import DiscoveryAgent
from cua.discovery.llm import LLMClient
from cua.discovery.recorder import CapabilityRecorder
from cua.escalation.broker import EscalationBroker
from cua.evidence.recorder import EvidenceRecorder
from cua.replay.engine import ReplayEngine
from cua.safety.policy import PolicyEngine, default_policy_for_demo
from cua.surface.web import open_web_surface

REPO = Path(__file__).resolve().parents[2]
CAPABILITIES = REPO / "capabilities"
EVIDENCE = REPO / "evidence"
TARGET = "http://127.0.0.1:8099"

# Credentials live here, never in an artifact, a prompt, or a log.
DEMO_SECRETS = {"operator_user": "demo_op", "operator_password": "demo_pass"}


def _runtime_conditions() -> tuple[list[BusinessOutcome], list[RecoverableCondition]]:
    """The declared error space for the Meridian Core flows.

    Written by hand, not inferred: a happy-path discovery run never sees these screens,
    so claiming the recorder discovered them would be a lie. This is the reviewed part
    of the capability, and the write-up is explicit that it is.
    """
    outcomes = [
        BusinessOutcome(
            code="MEMBER_NOT_FOUND",
            description="No member exists with the supplied number.",
            detect=StateAssertion(kind="text_present", value="No member found matching that ID"),
        ),
        BusinessOutcome(
            code="MEMBER_RECORD_NOT_FOUND",
            description="The member detail screen reports the record does not exist.",
            detect=StateAssertion(kind="text_present", value="Record not found"),
        ),
        BusinessOutcome(
            code="ACCESS_DENIED",
            description="The operator profile is not entitled to view this member.",
            detect=StateAssertion(kind="text_present", value="Access denied"),
        ),
        BusinessOutcome(
            code="VALIDATION_ERROR",
            description="The application rejected the supplied input.",
            detect=StateAssertion(kind="text_present", value="must be numeric"),
        ),
    ]
    recoverables = [
        RecoverableCondition(
            code="MAINTENANCE_INTERSTITIAL",
            description="A scheduled-maintenance notice appears before the content.",
            detect=StateAssertion(kind="text_present", value="Scheduled Maintenance Notice"),
            action=RecoveryAction.DISMISS,
            target=Target(
                description="Acknowledge button on the maintenance notice",
                frame="mainFrame",
                strategies=[
                    LocatorStrategy(tier=LocatorTier.ROLE_NAME,
                                    value={"role": "link", "name": "Acknowledge", "exact": True},
                                    note="The notice offers a single acknowledgement control."),
                ],
            ),
        ),
        RecoverableCondition(
            code="SESSION_EXPIRED",
            description="The session timed out mid-flow and the app returned to sign on.",
            detect=StateAssertion(kind="text_present", value="Your session has expired"),
            action=RecoveryAction.REAUTHENTICATE,
            max_attempts=1,
        ),
    ]
    return outcomes, recoverables


def cmd_discover(args) -> int:
    evidence = EvidenceRecorder(EVIDENCE, run_id=args.run_id or None)
    policy = PolicyEngine(default_policy_for_demo(TARGET))
    llm = LLMClient(model=args.model)
    surface = open_web_surface(headless=not args.headed, cdp_port=args.cdp_port)

    print(f"[discover] model={llm.model} goal={args.goal!r}")
    print(f"[discover] evidence -> {evidence.dir}")
    try:
        agent = DiscoveryAgent(surface, llm, policy, evidence, max_steps=args.max_steps)
        run = agent.run(goal=args.goal, entry_url=args.url, secrets=DEMO_SECRETS)
        evidence.write_json("discovery_run.json", run.model_dump())

        print(f"[discover] success={run.success} ({run.stopped_because})")
        for step in run.steps:
            mark = "ok " if step.result_ok else "FAIL"
            print(f"   {step.index:>2} {mark} {step.action.action:9} {str(step.action.ref or step.action.url or '')[:34]:36} {step.action.reasoning[:60]}")

        if not run.success:
            print("[discover] no artifact written: the run did not reach the goal")
            return 1

        outcomes, recoverables = _runtime_conditions()
        capability = CapabilityRecorder().record(
            run=run,
            capability_id=args.capability_id,
            name=args.name,
            description=args.goal,
            inputs=[
                Param(name="member_id", type="string", description="Member number to look up.",
                      sensitivity=Sensitivity.INTERNAL, example="12345", pattern=r"\d{3,10}"),
            ],
            outputs=[
                Output(
                    name="savings_balance", type="string",
                    description="Current savings share balance as displayed.",
                    sensitivity=Sensitivity.INTERNAL,
                    source=Target(
                        description="Savings balance cell on the member detail screen",
                        frame="mainFrame",
                        strategies=[
                            LocatorStrategy(
                                tier=LocatorTier.TABLE_CELL,
                                value={"row_contains": "Savings", "cell_index": 1},
                                note="Row located by its label, value read from the adjacent cell.",
                            )
                        ],
                    ),
                    transform="strip",
                ),
            ],
            checkpoint=StateAssertion(kind="text_present", value="Share Balances"),
            product="meridian-core",
            surface_kind=SurfaceKind.LEGACY_WEB,
            business_outcomes=outcomes,
            recoverables=recoverables,
            param_values={"member_id": args.member_id},
            evidence_ref=str(evidence.dir),
            secret_keys=list(DEMO_SECRETS),
        )
        # The sign-on prefix is what re-establishes an expired session.
        capability.reauth_step_ids = [
            s.id for s in capability.steps
            if (s.value or '').startswith('{{secret:') or s.target and s.target.description == 'Sign On'
        ]
        CAPABILITIES.mkdir(exist_ok=True)
        path = CAPABILITIES / f"{capability.id}.v{capability.version}.json"
        path.write_text(capability.model_dump_json(indent=2), encoding="utf-8", newline="")
        print(f"[discover] capability written -> {path}")
        return 0
    finally:
        surface.close()
        llm.close()


def cmd_replay(args) -> int:
    capability = Capability.model_validate_json(Path(args.capability).read_text(encoding="utf-8"))
    evidence = EvidenceRecorder(EVIDENCE, run_id=args.run_id or None)
    policy = PolicyEngine(default_policy_for_demo(TARGET), capability_id=capability.id)
    surface = open_web_surface(headless=not args.headed, cdp_port=args.cdp_port)
    broker = EscalationBroker(evidence, session_endpoint=
                              f"http://127.0.0.1:{args.cdp_port}" if args.cdp_port else None)
    try:
        engine = ReplayEngine(surface, policy, evidence, escalation=broker if args.escalate else None)
        params = json.loads(args.params) if args.params else {}
        result = engine.run(capability, params=params, tenant_id=args.tenant, secrets=DEMO_SECRETS)
        evidence.write_json("replay_result.json", result.model_dump())

        print(f"[replay] {capability.id} v{capability.version} params={params}")
        print(f"[replay] status={result.status.value}  {result.summary()}")
        for trace in result.trace:
            mark = "ok " if trace.ok else "FAIL"
            tier = f" tier={trace.matched_tier}" if trace.matched_tier else ""
            print(f"   {mark} {trace.step_id:16} {trace.action:9}{tier}  {trace.detail[:60]}")
        for rec in result.recoveries:
            print(f"   recovery: {rec.condition_code} via {rec.action} -> {'ok' if rec.succeeded else 'failed'}")
        print(f"[replay] evidence -> {evidence.dir}")
        return 0 if result.is_actionable_by_agent() else 2
    finally:
        surface.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cua")
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="LLM-driven discovery run; emits a capability")
    d.add_argument("--goal", required=True)
    d.add_argument("--url", default=f"{TARGET}/")
    d.add_argument("--member-id", default="12345")
    d.add_argument("--capability-id", default="meridian.member.read_savings_balance")
    d.add_argument("--name", default="Read member savings balance")
    d.add_argument("--model", default=None)
    d.add_argument("--max-steps", type=int, default=14)
    d.add_argument("--headed", action="store_true")
    d.add_argument("--cdp-port", type=int, default=None)
    d.add_argument("--run-id", default=None)
    d.set_defaults(func=cmd_discover)

    r = sub.add_parser("replay", help="deterministic replay of a saved capability")
    r.add_argument("--capability", required=True)
    r.add_argument("--params", default=None, help='JSON, e.g. {"member_id":"12345"}')
    r.add_argument("--tenant", default=None)
    r.add_argument("--escalate", action="store_true", help="escalate on unresolvable targets")
    r.add_argument("--headed", action="store_true")
    r.add_argument("--cdp-port", type=int, default=None)
    r.add_argument("--run-id", default=None)
    r.set_defaults(func=cmd_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
