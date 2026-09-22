"""
Turning a discovery run into a capability.

This is the hinge of the whole system, and the interesting work is *generalisation*.
A run is a sequence of concrete acts against concrete values: type "12345", click the
control that happened to be at [e7]. A capability has to be the reusable shape of that:
type {{member_id}}, click the control identified the way a human would identify it.

Three things happen here:

1. **Values become parameters.** A literal that matches a declared input is replaced by a
   binding. This is done by explicit declaration rather than by guessing which strings
   look like IDs - a recorder that infers parameters will eventually parameterise a
   branch code and nobody will notice until it runs against production.

2. **Elements become layered targets.** Each observed element yields every addressing
   strategy it can support, ordered by portability. This is where the robustness of
   replay is decided, so it is done from the element's own properties rather than by
   asking the model to invent a selector.

3. **The flow gets a contract.** Checkpoint, outputs, and a starter set of runtime
   conditions.

An honest limitation, stated here and in the write-up: a single happy-path run cannot
discover the error space. The recorder seeds the conditions it can reason about from the
target's known behaviour and marks the artifact as requiring review. Pretending otherwise
would mean shipping a capability whose declared outcomes are fiction.
"""

from __future__ import annotations

from datetime import datetime, timezone

from cua.artifact.schema import (
    ActionType, BusinessOutcome, Capability, LocatorStrategy, LocatorTier, Output,
    Param, RecoverableCondition, RecoveryAction, RiskClass, StateAssertion, Step,
    SurfaceRequirement, SurfaceKind, Target,
)
from cua.discovery.agent import DiscoveryRun
from cua.surface.base import Element


def target_from_element(element: Element | dict) -> Target:
    """Derive every addressing strategy an element can support, most portable first.

    The ordering is the robustness argument in code: role+name is how a person describes
    a control and is what a desktop AX API also gives you; the label anchor is what
    rescues legacy screens whose inputs have no accessible name; the form control name
    is stable in server-rendered apps but meaningless elsewhere; a DOM path is recorded
    only so there is a last resort, and is explicitly marked brittle.
    """
    data = element if isinstance(element, dict) else element.model_dump()
    role = data.get("role") or "button"
    name = (data.get("name") or "").strip()
    label = (data.get("label") or "").strip()
    text = (data.get("text") or "").strip()
    attrs = data.get("attrs") or {}
    frame = data.get("frame")

    strategies: list[LocatorStrategy] = []

    if name and role in {"button", "link", "checkbox", "radio", "combobox", "textbox"}:
        strategies.append(
            LocatorStrategy(
                tier=LocatorTier.ROLE_NAME,
                value={"role": role, "name": name, "exact": True},
                note="Role plus accessible name: portable across versions and surfaces.",
            )
        )
    if label:
        strategies.append(
            LocatorStrategy(
                tier=LocatorTier.LABEL_ANCHORED,
                value={"label": label, "exact": False},
                note="Anchored to the visible label; survives markup churn around the control.",
            )
        )
    if text and text != name and len(text) < 60 and role in {"link", "button"}:
        strategies.append(
            LocatorStrategy(
                tier=LocatorTier.TEXT_EXACT, value={"text": text},
                note="Visible text of the control.",
            )
        )
    if attrs.get("name"):
        strategies.append(
            LocatorStrategy(
                tier=LocatorTier.NAME_ATTR, value={"name": attrs["name"]},
                note="Form control name; stable in server-rendered apps, absent elsewhere.",
            )
        )
    if attrs.get("id") and not _looks_generated(attrs["id"]):
        strategies.append(
            LocatorStrategy(
                tier=LocatorTier.DOM_PATH, value={"selector": f"#{attrs['id']}"},
                note="Last resort. Recorded because the id looks author-written, not generated.",
            )
        )

    if not strategies:
        strategies.append(
            LocatorStrategy(
                tier=LocatorTier.TEXT_CONTAINS, value={"text": (name or text or label)[:40]},
                note="Weak fallback: no durable identifier was available on this control.",
            )
        )

    description = name or label or text or f"{role} control"
    return Target(description=description, frame=frame, strategies=strategies)


def _looks_generated(value: str) -> bool:
    """Reject ASP.NET-style positional ids like ctl00_MainContent_gv1_ctl02_lnkView.

    These look like stable identifiers and are the single most common trap in this class
    of application: the trailing index is the control's *position*, so the id silently
    changes when a row is inserted above it.
    """
    lowered = value.lower()
    return lowered.startswith("ctl") or "_ctl" in lowered or "$" in value


class CapabilityRecorder:
    def record(
        self,
        run: DiscoveryRun,
        capability_id: str,
        name: str,
        description: str,
        inputs: list[Param],
        outputs: list[Output],
        checkpoint: StateAssertion,
        product: str,
        surface_kind: SurfaceKind = SurfaceKind.LEGACY_WEB,
        business_outcomes: list[BusinessOutcome] | None = None,
        recoverables: list[RecoverableCondition] | None = None,
        param_values: dict[str, str] | None = None,
        evidence_ref: str | None = None,
        secret_keys: list[str] | None = None,
    ) -> Capability:
        param_values = param_values or {}
        secret_keys = secret_keys or []
        # value -> "{{param}}" so concrete literals become bindings
        reverse = {str(v): f"{{{{{k}}}}}" for k, v in param_values.items() if str(v)}
        # Models routinely hand back the *key* from a "{{secret:key}}" reference rather
        # than the reference itself. The real credential never reaches them - that part
        # of the boundary held - but the binding has to be restored or replay would type
        # the literal string "operator_password" into the password box.
        secret_refs = {key: f"{{{{secret:{key}}}}}" for key in secret_keys}

        steps: list[Step] = []
        frames_seen: set[str] = set()

        for entry in run.steps:
            action = entry.action
            if action.action in {"done", "give_up"} or not entry.result_ok:
                continue

            snapshot = entry.element_snapshot
            if snapshot and snapshot.get("frame"):
                frames_seen.add(snapshot["frame"])

            step_id = f"s{len(steps) + 1:02d}_{action.action}"
            target = target_from_element(snapshot) if snapshot else None
            value = action.value
            if value and value in secret_refs:
                value = secret_refs[value]
            elif value and value.strip().startswith("{{secret:"):
                value = value.strip()
            elif value and value in reverse:
                value = reverse[value]

            risk = self._risk_for(action.action, target.description if target else "")

            steps.append(
                Step(
                    id=step_id,
                    action=ActionType(action.action),
                    description=action.reasoning[:200] or f"{action.action} {target.description if target else ''}",
                    target=target,
                    value=value,
                    url=action.url,
                    risk=risk,
                )
            )

        capability = Capability(
            id=capability_id,
            name=name,
            description=description,
            surface=SurfaceRequirement(
                kind=surface_kind, product=product, entry_url=run.entry_url,
                frames_expected=sorted(frames_seen),
            ),
            inputs=inputs,
            outputs=outputs,
            business_outcomes=business_outcomes or [],
            recoverables=recoverables or [],
            steps=steps,
            checkpoint=checkpoint,
            recorded_from_goal=run.goal,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            recorded_by_model=run.model,
            evidence_ref=evidence_ref,
        )
        return capability

    @staticmethod
    def _risk_for(action: str, description: str) -> RiskClass:
        """Mirror of the policy classifier, applied at record time.

        Recording the risk into the artifact means a reviewer sees the classification
        before the capability is ever invoked, rather than discovering it at runtime.
        """
        from cua.safety.policy import PolicyEngine, PolicyConfig

        if action in {"navigate", "extract", "fill", "select"}:
            return RiskClass.READ
        return PolicyEngine(PolicyConfig()).classify("button", description)
