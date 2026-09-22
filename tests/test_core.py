"""
Tests for the decisions, not the plumbing.

These run in about a second with no browser and no model, because they cover the parts
where a mistake is silent and expensive: risk classification, redaction, the control
lease, tenant overlay resolution, and the schema invariants that stop a malformed
capability from ever reaching the replay engine.

The browser-level behaviour is covered by scripts/demo_matrix.py, which exercises all
eight runtime conditions against the live app; duplicating that here in mocks would
test the mocks.
"""

from __future__ import annotations

import time

import pytest

from cua.artifact.schema import (
    ActionType, Capability, LocatorStrategy, LocatorTier, Param, RiskClass,
    Sensitivity, StateAssertion, Step, SurfaceKind, SurfaceRequirement, Target,
    TenantOverlay,
)
from cua.escalation.broker import Controller, EscalationBroker
from cua.evidence.recorder import EvidenceRecorder, redact
from cua.safety.policy import PolicyConfig, PolicyEngine, default_policy_for_demo


# --- helpers -----------------------------------------------------------------

def _target(desc="Search", tier=LocatorTier.ROLE_NAME, **value) -> Target:
    return Target(
        description=desc,
        strategies=[LocatorStrategy(tier=tier, value=value or {"role": "button", "name": desc})],
    )


def _capability(**overrides) -> Capability:
    base = dict(
        id="test.cap", name="Test", description="A test capability",
        surface=SurfaceRequirement(kind=SurfaceKind.LEGACY_WEB, product="meridian-core"),
        inputs=[Param(name="member_id", type="string", description="id")],
        steps=[
            Step(id="s1", action=ActionType.FILL, description="type the id",
                 target=_target("Member ID"), value="{{member_id}}"),
        ],
        checkpoint=StateAssertion(kind="text_present", value="Share Balances"),
    )
    base.update(overrides)
    return Capability(**base)


# --- schema invariants -------------------------------------------------------

def test_locator_strategies_are_sorted_by_portability():
    target = Target(
        description="Search",
        strategies=[
            LocatorStrategy(tier=LocatorTier.DOM_PATH, value={"selector": "#x"}),
            LocatorStrategy(tier=LocatorTier.ROLE_NAME, value={"role": "button", "name": "Search"}),
            LocatorStrategy(tier=LocatorTier.NAME_ATTR, value={"name": "btnSearch"}),
        ],
    )
    # Replay tries these in order, so the ordering is a correctness property, not cosmetics.
    assert [s.tier for s in target.strategies] == [
        LocatorTier.ROLE_NAME, LocatorTier.NAME_ATTR, LocatorTier.DOM_PATH
    ]


def test_step_referencing_an_undeclared_input_is_rejected():
    with pytest.raises(ValueError, match="undeclared input"):
        _capability(steps=[
            Step(id="s1", action=ActionType.FILL, description="x",
                 target=_target(), value="{{not_declared}}")
        ])


def test_secret_reference_is_not_treated_as_an_input():
    # Credentials are a separate binding namespace: an input is part of the public
    # contract the calling agent fills in, and a password must never be.
    cap = _capability(steps=[
        Step(id="s1", action=ActionType.FILL, description="password",
             target=_target(), value="{{secret:operator_password}}")
    ])
    assert cap.steps[0].value == "{{secret:operator_password}}"


def test_duplicate_step_ids_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        _capability(steps=[
            Step(id="dup", action=ActionType.CLICK, description="a", target=_target()),
            Step(id="dup", action=ActionType.CLICK, description="b", target=_target()),
        ])


def test_capability_max_risk_is_raised_to_its_riskiest_step():
    cap = _capability(steps=[
        Step(id="s1", action=ActionType.CLICK, description="read", target=_target(), risk=RiskClass.READ),
        Step(id="s2", action=ActionType.CLICK, description="post", target=_target(),
             risk=RiskClass.IRREVERSIBLE),
    ])
    # A reviewer gating on the capability must not be able to miss a buried risky step.
    assert cap.max_risk is RiskClass.IRREVERSIBLE


def test_navigate_step_requires_a_url():
    with pytest.raises(ValueError, match="navigate step requires url"):
        Step(id="s1", action=ActionType.NAVIGATE, description="go")


# --- policy ------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("Search", RiskClass.READ),
        ("btnSearch", RiskClass.READ),            # camelCase legacy control name
        ("lnkViewDetail", RiskClass.READ),
        ("Open Account", RiskClass.WRITE),
        ("btnOpenAccount", RiskClass.WRITE),
        ("Transfer Funds", RiskClass.IRREVERSIBLE),
        ("cmdPostTransaction", RiskClass.IRREVERSIBLE),
        ("Delete Member", RiskClass.IRREVERSIBLE),
        ("btnWhoKnows", RiskClass.WRITE),          # unrecognised -> fail closed
    ],
)
def test_risk_classification(name, expected):
    assert PolicyEngine(PolicyConfig()).classify("button", name) is expected


def test_irreversible_actions_are_never_auto_approved_even_when_capability_is_trusted():
    policy = PolicyEngine(
        PolicyConfig(confirmed_write_capabilities=["test.cap"], max_autonomous_risk=RiskClass.WRITE),
        capability_id="test.cap",
    )
    decision = policy.check_action("click", "button", "Transfer Funds")
    assert not decision.allowed
    assert decision.requires_confirmation


def test_allowlist_blocks_other_origins_and_denied_paths():
    policy = PolicyEngine(default_policy_for_demo("http://127.0.0.1:8099"))
    assert policy.check_navigation("http://127.0.0.1:8099/content/search").allowed
    assert not policy.check_navigation("http://evil.example.com/").allowed
    # The fault switchboard is a test affordance; automation must never reach it.
    assert not policy.check_navigation("http://127.0.0.1:8099/admin/faults").allowed


# --- redaction ---------------------------------------------------------------

def test_secrets_and_pii_are_scrubbed_before_anything_is_written():
    cleaned = redact({
        "password": "hunter2",
        "note": "email bob@example.com, ssn 123-45-6789, card 4111111111111111",
        "nested": [{"token": "abc123"}],
    })
    assert cleaned["password"] == "[REDACTED]"
    assert "bob@example.com" not in cleaned["note"]
    assert "123-45-6789" not in cleaned["note"]
    assert "4111111111111111" not in cleaned["note"]
    assert cleaned["nested"][0]["token"] == "[REDACTED]"


def test_evidence_log_redacts_on_write(tmp_path):
    rec = EvidenceRecorder(tmp_path, run_id="r1")
    rec.log("act", {"password": "hunter2", "member": "ssn 123-45-6789"})
    written = (tmp_path / "r1" / "events.jsonl").read_text(encoding="utf-8")
    # Redaction happens at the write boundary: a log that holds PII until some later
    # scrubbing pass has already leaked it into every backup taken in between.
    assert "hunter2" not in written
    assert "123-45-6789" not in written


# --- control transfer --------------------------------------------------------

def test_lease_moves_automation_to_operator_and_back(tmp_path):
    broker = EscalationBroker(EvidenceRecorder(tmp_path, run_id="esc"))
    assert broker.who_is_in_control() is Controller.AUTOMATION

    request = broker.raise_intervention(
        capability_id="test.cap", goal="g", step_id="s7", reason="needs a human"
    )
    # Automation gives up the session the moment it asks for help; otherwise two actors
    # can act on the same screen.
    assert broker.who_is_in_control() is Controller.NOBODY
    assert not broker.automation_may_act()

    broker.claim(request.id, "alice")
    assert broker.who_is_in_control() is Controller.OPERATOR

    broker.record_human_action(request.id, "clicked View")
    broker.release(request.id)
    assert broker.who_is_in_control() is Controller.AUTOMATION
    assert len(broker.requests[request.id].human_actions) == 1


def test_second_operator_cannot_take_an_already_claimed_intervention(tmp_path):
    broker = EscalationBroker(EvidenceRecorder(tmp_path, run_id="esc2"))
    request = broker.raise_intervention(capability_id="c", goal="g", step_id="s", reason="r")
    broker.claim(request.id, "alice")
    with pytest.raises(RuntimeError):
        broker.claim(request.id, "bob")
    assert broker.lease.holder_id == "alice"


def test_a_held_session_blocks_claiming_a_different_intervention(tmp_path):
    """The lease guards the *session*, not just one ticket.

    Two interventions can exist against the same session; only one person may be
    driving it. This is the path the state check above does not cover.
    """
    broker = EscalationBroker(EvidenceRecorder(tmp_path, run_id="esc2b"))
    first = broker.raise_intervention(capability_id="c", goal="g", step_id="s1", reason="r")
    second = broker.raise_intervention(capability_id="c", goal="g", step_id="s2", reason="r")
    broker.claim(first.id, "alice")
    with pytest.raises(RuntimeError, match="already held"):
        broker.claim(second.id, "bob")


def test_abandoned_lease_is_reclaimed_so_a_session_is_never_stranded(tmp_path):
    broker = EscalationBroker(EvidenceRecorder(tmp_path, run_id="esc3"), operator_lease_seconds=0)
    request = broker.raise_intervention(capability_id="c", goal="g", step_id="s", reason="r")
    broker.claim(request.id, "alice")
    time.sleep(0.01)
    assert broker.reclaim_if_expired()
    assert broker.who_is_in_control() is Controller.NOBODY


# --- multi-tenant ------------------------------------------------------------

def test_tenant_overlay_specialises_without_mutating_the_base_capability():
    base = _capability(
        steps=[
            Step(id="s1", action=ActionType.CLICK, description="search", target=_target("Search")),
            Step(id="s2", action=ActionType.CLICK, description="view", target=_target("View")),
        ],
        tenant_overlays={
            "cu_north": TenantOverlay(
                tenant_id="cu_north",
                entry_url="https://north.example/core/",
                target_overrides={"s1": _target("Find Member")},
                extra_steps_after={
                    "s1": [Step(id="s1b", action=ActionType.CLICK, description="extra consent",
                                target=_target("Continue"))]
                },
            )
        },
    )

    resolved = base.resolved_for("cu_north")
    assert resolved.surface.entry_url == "https://north.example/core/"
    assert [s.id for s in resolved.steps] == ["s1", "s1b", "s2"]
    assert resolved.steps[0].target.description == "Find Member"

    # The base is untouched, so a reviewer can diff base against resolved and see
    # exactly what one tenant changed.
    assert base.surface.entry_url is None
    assert [s.id for s in base.steps] == ["s1", "s2"]
    assert base.steps[0].target.description == "Search"


def test_unknown_tenant_falls_back_to_the_shared_flow():
    base = _capability()
    assert base.resolved_for("never_onboarded") is base
