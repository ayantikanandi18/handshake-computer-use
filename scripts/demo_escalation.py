"""
Demonstrates the human handoff on a live session.

The point being proved here is the one the brief cares about: the operator works in the
*same* browser session the automation was driving, not a fresh one. So the script does
not simulate a handoff - it performs one:

  1. replay runs and hits a step it is not permitted to take alone,
  2. automation raises an intervention and drops the session lease,
  3. an "operator" attaches to the same live browser over CDP and does the work by hand,
  4. what the operator did is recorded,
  5. the lease returns to automation and the run continues on the state the human left.

The operator console is a print statement, which the brief expressly allows. The
mechanism underneath it - lease transfer, CDP attach, audit of human actions, resume on
the same page - is real.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cua.artifact.schema import Capability
from cua.cli import DEMO_SECRETS, TARGET
from cua.escalation.broker import Controller, EscalationBroker
from cua.evidence.recorder import EvidenceRecorder
from cua.replay.engine import ReplayEngine
from cua.safety.policy import PolicyConfig, PolicyEngine
from cua.artifact.schema import RiskClass
from cua.surface.web import open_web_surface

REPO = Path(__file__).resolve().parents[1]
CAPABILITY = REPO / "capabilities" / "meridian.member.read_savings_balance.v1.json"
CDP_PORT = 9333


def main() -> int:
    capability = Capability.model_validate_json(CAPABILITY.read_text(encoding="utf-8"))

    # Force the escalation: this policy forbids the click that opens the member record,
    # which is exactly the shape of "a risky step needs a person to decide".
    policy = PolicyEngine(
        PolicyConfig(
            allowed_origins=[TARGET],
            allowed_path_prefixes=["/"],
            denied_path_patterns=[r"^/admin/"],
            max_autonomous_risk=RiskClass.READ,
        ),
        capability_id=capability.id,
    )

    evidence = EvidenceRecorder(REPO / "evidence", run_id="escalation-demo")
    surface = open_web_surface(headless=True, cdp_port=CDP_PORT)
    broker = EscalationBroker(
        evidence, session_endpoint=f"http://127.0.0.1:{CDP_PORT}"
    )

    try:
        # Make the "View" step risky so the run must stop and ask.
        for step in capability.steps:
            if step.target and step.target.description == "View":
                step.risk = RiskClass.WRITE

        engine = ReplayEngine(surface, policy, evidence, escalation=broker)
        print(f"control before run: {broker.who_is_in_control().value}")

        result = engine.run(capability, params={"member_id": "12345"}, secrets=DEMO_SECRETS)
        print(f"replay stopped   : {result.status.value} ({result.observed})")
        print(f"control now      : {broker.who_is_in_control().value}")

        if result.intervention_id is None:
            print("no intervention was raised; nothing to hand off")
            return 1

        request = broker.requests[result.intervention_id]
        print()
        print("--- operator console (mocked UI, real handoff) ---")
        print(f"  intervention : {request.id}")
        print(f"  capability   : {request.capability_id}")
        print(f"  stopped at   : {request.step_id}")
        print(f"  reason       : {request.reason[:90]}")
        print(f"  screenshot   : {request.screenshot_path}")
        print(f"  live session : {request.session_endpoint}")

        # An operator claims the work and takes the lease.
        broker.claim(request.id, operator_id="alice.operator")
        print(f"  claimed by   : alice.operator -> control = {broker.who_is_in_control().value}")

        # The operator attaches to the SAME browser over CDP and finishes the step by
        # hand. This is the part that cannot be faked: a different browser would not
        # have the authenticated session or the search results on screen.
        # Attach through the surface's own driver: a second sync Playwright driver in
        # the same thread is not permitted, and more importantly the operator must land
        # on the *existing* browser, not a new one.
        operator_browser = surface.attach_operator(f"http://127.0.0.1:{CDP_PORT}")
        try:
            context = operator_browser.contexts[0]
            page = context.pages[0]
            print(f"  operator sees: {page.url}")
            main_frame = next(
                (f for f in page.frames if f.name == "mainFrame"), page.main_frame
            )
            main_frame.get_by_role("link", name="View").first.click()
            page.wait_for_load_state("load")
            broker.record_human_action(
                request.id,
                "Clicked View on member 12345 after confirming the request was legitimate",
                location=page.url,
            )
            print("  operator did : clicked View on the result row")
        finally:
            operator_browser.close()

        # Hand control back; automation resumes on the state the human left behind.
        broker.release(request.id, resume_instruction="member detail is open, continue")
        print(f"  released     -> control = {broker.who_is_in_control().value}")

        observation = surface.observe()
        on_detail = "Share Balances" in observation.text
        print()
        print(f"automation resumes on the operator's page: Share Balances visible = {on_detail}")

        outputs, missing = engine._extract_outputs(capability, observation)
        print(f"outputs after handoff: {outputs} (missing: {missing})")

        evidence.write_json(
            "escalation_demo.json",
            {
                "intervention": request.model_dump(),
                "resumed_on_detail_screen": on_detail,
                "outputs": outputs,
                "final_controller": broker.who_is_in_control().value,
            },
        )
        return 0 if on_detail and outputs else 1
    finally:
        surface.close()


if __name__ == "__main__":
    raise SystemExit(main())
