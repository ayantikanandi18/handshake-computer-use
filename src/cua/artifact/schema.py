"""
The capability artifact: the contract between a discovery run and everything downstream.

This schema is the centre of the system, so the reasoning behind it is worth stating.

A recorded flow has to serve three different readers, and they want different things:

  * the calling agent wants a *typed function signature* - what do I pass in, what do I
    get back, and what can legitimately go wrong,
  * the replay engine wants *executable detail* - exactly how to find each control and
    how to tell whether the step worked,
  * a human reviewer wants to answer "is this safe to let loose on our core system?"
    without reading a model transcript.

So the artifact is deliberately not a transcript and not a raw action tape. It is a
declaration: inputs, outputs, the closed set of business outcomes, the steps, and the
checkpoint that defines success.

Two choices here are load-bearing and are defended in the write-up:

1. Targets are recorded as an *ordered list of strategies*, not one selector. Legacy bank
   UIs have no test IDs and their generated control names (ctl00_MainContent_gv1_ctl02_...)
   look stable but are positional. A single selector is a single point of failure, and the
   tier that actually matched at replay time is the cheapest drift signal available.

2. Business outcomes are *declared in the artifact*, not inferred at runtime. "No such
   member" is a legitimate answer to a question, not a crash. If the calling agent is to
   branch on it, the set has to be closed, typed, and reviewable up front.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"


# --- how we find a control ---------------------------------------------------

class LocatorTier(str, Enum):
    """Preference order for finding a control. Lower tiers are more portable.

    The ordering is a portability judgement, not a DOM judgement: ROLE_NAME and
    LABEL_ANCHORED describe a control the way a human operator would describe it
    ("the Search button", "the box next to Member ID"), which is why they survive a
    reskin, a version bump, and - importantly - the jump to a desktop surface where
    there is no DOM at all. DOM_PATH is last because it is the first thing to break.
    """

    ROLE_NAME = "role_name"            # accessible role + name: portable across surfaces
    LABEL_ANCHORED = "label_anchored"  # the control next to this label text
    TABLE_CELL = "table_cell"          # row identified by key text, column by header
    TEXT_EXACT = "text_exact"
    TEXT_CONTAINS = "text_contains"
    NAME_ATTR = "name_attr"            # form control name= (stable in server-rendered apps)
    DOM_PATH = "dom_path"              # brittle, recorded only as a last resort


class LocatorStrategy(BaseModel):
    """One way to find a control. Several of these make up a Target."""

    tier: LocatorTier
    # Interpretation depends on tier, kept loose on purpose so a desktop or terminal
    # surface can carry its own addressing (automation id, control id, screen cell)
    # without changing the schema.
    value: dict[str, Any]
    note: str | None = Field(
        default=None, description="Why this strategy was recorded; shown to reviewers."
    )


class Target(BaseModel):
    """A control, described several ways, most portable first."""

    description: str = Field(description="Human-readable: 'the Member ID input'.")
    frame: str | None = Field(
        default=None,
        description="Frame name for frameset apps. None means the top-level document.",
    )
    strategies: list[LocatorStrategy] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered_by_tier(self) -> "Target":
        order = list(LocatorTier)
        self.strategies.sort(key=lambda s: order.index(s.tier))
        return self


# --- typed contract ----------------------------------------------------------

class Sensitivity(str, Enum):
    """Drives redaction. Regulated data must never reach an artifact or a log."""

    PUBLIC = "public"
    INTERNAL = "internal"
    PII = "pii"           # member name, address, DOB
    SECRET = "secret"     # credentials, tokens - never persisted anywhere, ever


class Param(BaseModel):
    name: str
    type: Literal["string", "integer", "number", "boolean"]
    description: str
    required: bool = True
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    example: Any | None = None
    pattern: str | None = Field(default=None, description="Optional validation regex.")


class Output(BaseModel):
    name: str
    type: Literal["string", "integer", "number", "boolean"]
    description: str
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    # Where the value comes from. Extraction is declared, not inferred at replay time.
    source: Target
    extract: Literal["text", "value", "attribute"] = "text"
    attribute: str | None = None
    transform: Literal["none", "strip", "currency_to_number"] = "none"


# --- outcomes ----------------------------------------------------------------

class BusinessOutcome(BaseModel):
    """A legitimate non-success answer the caller must be able to branch on.

    Declaring these is what keeps "no such member" from being reported as a failure.
    The detector is a state signature rather than an exception: the app tells us in
    the UI, so we read it the same way an operator would.
    """

    code: str = Field(description="Stable machine code, e.g. MEMBER_NOT_FOUND.")
    description: str
    detect: "StateAssertion"
    terminal: bool = Field(
        default=True, description="If true, replay stops here and reports this outcome."
    )


class RecoveryAction(str, Enum):
    DISMISS = "dismiss"          # click through a known interstitial
    WAIT_RETRY = "wait_retry"    # transient slowness
    REAUTHENTICATE = "reauth"    # session expired mid-flow
    ESCALATE = "escalate"        # hand to a human


class RecoverableCondition(BaseModel):
    """A runtime condition replay is allowed to handle by itself, within limits."""

    code: str
    description: str
    detect: "StateAssertion"
    action: RecoveryAction
    target: Target | None = Field(default=None, description="Control to click for DISMISS.")
    max_attempts: int = Field(default=2, ge=1, le=5)


# --- state assertions --------------------------------------------------------

class StateAssertion(BaseModel):
    """How we tell what screen we are on and whether a step did what it should.

    Replay is only deterministic if every step declares what it expects to be true
    afterwards. Without this, replay is a blind action tape that happily types a member
    ID into a login box.
    """

    kind: Literal["text_present", "text_absent", "url_matches", "element_present", "http_status"]
    value: str
    frame: str | None = None
    case_sensitive: bool = False


# --- steps -------------------------------------------------------------------

class RiskClass(str, Enum):
    """Whether a step changes anything, and how hard it is to undo.

    This is deliberately a property of the *step*, not the capability: a flow that reads
    a balance and a flow that opens an account share most of their steps, and the policy
    engine needs to reason at the granularity where the risk actually lives.
    """

    READ = "read"                  # navigation, reading, searching: reversible by definition
    WRITE = "write"                # creates or changes a record; undoable with effort
    IRREVERSIBLE = "irreversible"  # money movement, external notification, deletion


class ActionType(str, Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    SELECT = "select"
    PRESS = "press"
    WAIT_FOR = "wait_for"
    EXTRACT = "extract"
    ASSERT = "assert"


class Step(BaseModel):
    id: str
    action: ActionType
    description: str = Field(description="Reviewer-facing: what this step does and why.")
    target: Target | None = None
    # Literal, or "{{ param_name }}" to bind an input at invocation time.
    value: str | None = None
    url: str | None = Field(default=None, description="For NAVIGATE.")
    risk: RiskClass = RiskClass.READ
    expect: StateAssertion | None = Field(
        default=None, description="What must be true after this step for it to count."
    )
    timeout_ms: int = 10_000
    optional: bool = Field(
        default=False,
        description="If true, a miss is logged and skipped rather than failing the run.",
    )

    @model_validator(mode="after")
    def _action_shape(self) -> "Step":
        if self.action is ActionType.NAVIGATE and not self.url:
            raise ValueError("navigate step requires url")
        needs_target = {
            ActionType.CLICK, ActionType.FILL, ActionType.SELECT,
            ActionType.EXTRACT, ActionType.WAIT_FOR,
        }
        if self.action in needs_target and self.target is None:
            raise ValueError(f"{self.action} step requires a target")
        if self.action in {ActionType.FILL, ActionType.SELECT} and self.value is None:
            raise ValueError(f"{self.action} step requires a value")
        return self


# --- surface + tenancy -------------------------------------------------------

class SurfaceKind(str, Enum):
    WEB = "web"
    LEGACY_WEB = "legacy_web"   # framesets, table layout, no test IDs
    DESKTOP = "desktop"
    TERMINAL = "terminal"       # 3270/5250 green screen, still very much alive


class SurfaceRequirement(BaseModel):
    """What the flow assumes about the surface it runs against.

    Replay refuses to run against a surface that does not satisfy this, which is what
    stops a capability recorded on one tenant's build from silently half-working on
    another's.
    """

    kind: SurfaceKind
    product: str = Field(description="Vendor product id, e.g. 'meridian-core'.")
    version_range: str = Field(default="*", description="Semver-ish range the flow was recorded against.")
    entry_url: str | None = None
    frames_expected: list[str] = Field(default_factory=list)


class TenantOverlay(BaseModel):
    """Per-tenant specialisation of a shared capability.

    Hundreds of tenants run the same vendor product with different branding, config and
    versions. Re-recording per tenant does not scale and loses the shared improvement.
    An overlay keeps one base flow and expresses only the delta: a replaced target, a
    changed entry point, an extra step some tenants need.
    """

    tenant_id: str
    entry_url: str | None = None
    target_overrides: dict[str, Target] = Field(
        default_factory=dict, description="step_id -> replacement target."
    )
    value_overrides: dict[str, str] = Field(default_factory=dict)
    extra_steps_after: dict[str, list[Step]] = Field(
        default_factory=dict, description="step_id -> steps to insert after it."
    )
    notes: str | None = None


# --- the capability ----------------------------------------------------------

class Capability(BaseModel):
    """A recorded, reusable, agent-invocable flow."""

    schema_version: str = SCHEMA_VERSION
    id: str = Field(description="Stable id, e.g. 'meridian.member.read_savings_balance'.")
    version: int = Field(default=1, ge=1)
    name: str
    description: str = Field(description="What an agent should call this for.")

    surface: SurfaceRequirement
    inputs: list[Param] = Field(default_factory=list)
    outputs: list[Output] = Field(default_factory=list)
    business_outcomes: list[BusinessOutcome] = Field(default_factory=list)
    recoverables: list[RecoverableCondition] = Field(default_factory=list)

    steps: list[Step]
    checkpoint: StateAssertion = Field(
        description="The single condition that means the goal was actually achieved."
    )

    max_risk: RiskClass = Field(
        default=RiskClass.READ,
        description="Highest risk class any step carries; lets policy gate at the capability level.",
    )

    reauth_step_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Steps that re-establish a session, in order. Declared rather than inferred: "
            "the engine used to guess the login prefix from step ids, which silently "
            "stopped working the moment the recorder named steps s01_fill instead of "
            "login_user. Session recovery is too important to rest on a naming convention."
        ),
    )

    # Provenance. Deliberately excludes the model transcript: the artifact is the
    # contract, the transcript is evidence, and mixing them tempts replay to depend on
    # the model's prose.
    recorded_from_goal: str | None = None
    recorded_at: str | None = None
    recorded_by_model: str | None = None
    evidence_ref: str | None = Field(default=None, description="Path to the discovery run evidence.")

    tenant_overlays: dict[str, TenantOverlay] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistency(self) -> "Capability":
        ids = [s.id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        declared = {p.name for p in self.inputs}
        for step in self.steps:
            value = (step.value or "").strip()
            # "{{secret:name}}" is a different binding namespace from "{{param}}".
            # Secrets are resolved from the runtime secret store and are deliberately
            # NOT declared as inputs: an input is part of the public contract the
            # calling agent supplies, and a credential must never be either.
            if value.startswith("{{secret:"):
                continue
            if value.startswith("{{"):
                ref = value.strip("{} ").strip()
                if ref not in declared:
                    raise ValueError(f"step {step.id} references undeclared input '{ref}'")
        order = list(RiskClass)
        highest = max((s.risk for s in self.steps), key=lambda r: order.index(r), default=RiskClass.READ)
        if order.index(highest) > order.index(self.max_risk):
            self.max_risk = highest
        return self

    def resolved_for(self, tenant_id: str | None) -> "Capability":
        """Apply a tenant overlay, returning a flow ready to execute.

        Kept as a pure function so the base capability is never mutated and a reviewer
        can diff base against resolved to see exactly what a tenant changed.
        """
        if not tenant_id or tenant_id not in self.tenant_overlays:
            return self
        overlay = self.tenant_overlays[tenant_id]
        clone = self.model_copy(deep=True)
        if overlay.entry_url:
            clone.surface.entry_url = overlay.entry_url

        steps: list[Step] = []
        for step in clone.steps:
            if step.id in overlay.target_overrides:
                step.target = overlay.target_overrides[step.id]
            if step.id in overlay.value_overrides:
                step.value = overlay.value_overrides[step.id]
            steps.append(step)
            steps.extend(overlay.extra_steps_after.get(step.id, []))
        clone.steps = steps
        return clone


BusinessOutcome.model_rebuild()
RecoverableCondition.model_rebuild()
