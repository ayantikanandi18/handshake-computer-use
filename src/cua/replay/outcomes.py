"""
The result contract for a replay.

The brief asks for three things to be distinguishable, and getting this taxonomy right
matters more than almost anything else in the system, because it is what the calling
agent branches on.

  SUCCESS            the flow completed and the checkpoint held; outputs returned
  BUSINESS_OUTCOME   a legitimate answer that is not success ("no such member")
  FAILURE            the automation could not complete and a human needs to know

The distinction that is easy to get wrong is the second one. "Member not found" is not
an error - it is the answer to the question that was asked. If replay raises on it, every
caller has to parse exception text to recover a fact the UI stated plainly, and the agent
above cannot reason about it. So business outcomes are declared in the artifact, detected
deliberately, and returned as data.

Recovery is not a result type. It is something that happens *during* a run - dismissing
an interstitial, waiting out a slow load, re-authenticating - and is reported in
`recoveries` alongside whatever the run eventually produced. A run that recovered and
then succeeded is a success, but the recovery is worth surfacing because a capability
that recovers on every invocation is telling you something is wrong.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ResultStatus(str, Enum):
    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    FAILURE = "failure"
    ESCALATED = "escalated"


class FailureKind(str, Enum):
    """Why a run could not finish. Each maps to a different operational response."""

    TARGET_NOT_FOUND = "target_not_found"        # a control the flow needs is gone
    AMBIGUOUS_TARGET = "ambiguous_target"        # matched several controls; refused to guess
    ASSERTION_FAILED = "assertion_failed"        # step ran but the screen is not what we expect
    CHECKPOINT_FAILED = "checkpoint_failed"      # flow ran to the end but did not achieve the goal
    TIMEOUT = "timeout"
    POLICY_BLOCKED = "policy_blocked"            # the flow wanted to do something not permitted
    SURFACE_ERROR = "surface_error"              # app 500, browser crash, navigation error
    SESSION_LOST = "session_lost"                # auth expired and could not be recovered
    OUTPUT_EXTRACTION_FAILED = "output_extraction_failed"
    UNRECOVERABLE_CONDITION = "unrecoverable_condition"


class RecoveryRecord(BaseModel):
    condition_code: str
    action: str
    attempts: int
    succeeded: bool
    at_step: str


class StepTrace(BaseModel):
    """Per-step record. This is what makes a failure debuggable rather than mysterious."""

    step_id: str
    action: str
    description: str = ""
    ok: bool
    detail: str = ""
    # Which locator tier actually matched. The single most useful drift signal the
    # system produces: a flow silently sliding down the tiers is a flow about to break.
    matched_tier: str | None = None
    expected: str | None = None
    observed: str | None = None
    duration_ms: int = 0


class ReplayResult(BaseModel):
    status: ResultStatus
    capability_id: str
    capability_version: int
    tenant_id: str | None = None

    outputs: dict[str, Any] = Field(default_factory=dict)

    # Present when status is BUSINESS_OUTCOME.
    outcome_code: str | None = None
    outcome_detail: str | None = None

    # Present when status is FAILURE.
    failure_kind: FailureKind | None = None
    failed_step: str | None = None
    expected: str | None = None
    observed: str | None = None
    evidence_dir: str | None = None

    # Present when status is ESCALATED.
    intervention_id: str | None = None

    recoveries: list[RecoveryRecord] = Field(default_factory=list)
    trace: list[StepTrace] = Field(default_factory=list)
    duration_ms: int = 0

    def is_actionable_by_agent(self) -> bool:
        """True when the calling agent got an answer it can reason about."""
        return self.status in {ResultStatus.SUCCESS, ResultStatus.BUSINESS_OUTCOME}

    def summary(self) -> str:
        if self.status is ResultStatus.SUCCESS:
            return f"success: {self.outputs}"
        if self.status is ResultStatus.BUSINESS_OUTCOME:
            return f"business outcome {self.outcome_code}: {self.outcome_detail}"
        if self.status is ResultStatus.ESCALATED:
            return f"escalated to a human (intervention {self.intervention_id})"
        return (
            f"failure {self.failure_kind.value if self.failure_kind else '?'} "
            f"at step {self.failed_step}: expected {self.expected!r}, observed {self.observed!r}"
        )
