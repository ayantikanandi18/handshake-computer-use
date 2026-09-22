"""
Human-in-the-loop: pausing automation and handing a live session to a person.

The requirement that shapes this module is "the same live session, not a fresh one".
That rules out the easy implementation - screenshot the failure, open a ticket, let a
human redo the work somewhere else - because by the time a person starts over, the
session state that made the problem interesting is gone. Half of a back-office flow is
stateful: a search result set, a half-filled form, an authenticated session with a
timeout running.

So control is modelled as a **lease** on a session. Exactly one holder at a time, and
the transfer is explicit in both directions:

    AUTOMATION --(raise_intervention)--> PENDING --(claim)--> OPERATOR
         ^                                                        |
         +---------------------(release)--------------------------+

Why a lease rather than a lock or a flag:

* It answers "who is in control right now" as a first-class question, which the brief
  explicitly asks for. A boolean `paused` does not, and nothing stops two actors acting
  on a paused session.
* It is observable. The lease holder, the reason, and the timestamps are the audit trail
  for a regulated environment: who touched the member record, when, and why.
* It has a timeout. An operator who wanders off must not strand the session forever, so
  an expired lease is reclaimable and that reclamation is itself an event.

The operator console is deliberately minimal - the brief permits mocking it. What is
*not* mocked is the mechanism: the browser is launched with a debugging port, so the
operator attaches to the very same browser the automation was driving, and anything they
do is visible to the automation when it resumes.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Controller(str, Enum):
    AUTOMATION = "automation"
    OPERATOR = "operator"
    NOBODY = "nobody"


class InterventionState(str, Enum):
    PENDING = "pending"       # raised, nobody has picked it up
    CLAIMED = "claimed"       # an operator holds the lease
    RESOLVED = "resolved"     # handed back, run may continue
    ABANDONED = "abandoned"   # lease expired or operator gave up


class HumanAction(BaseModel):
    """What the operator did, captured for audit and for improving the capability."""

    at: str
    description: str
    location: str | None = None


class InterventionRequest(BaseModel):
    """Everything a human needs to act, carried with the request.

    The context is not a courtesy. An operator picking this up has no memory of the run,
    so the request has to answer: what was this trying to do, where did it stop, what is
    on screen, and why did it stop. A ticket that just says "automation failed" spends a
    person's attention on reconstruction.
    """

    id: str = Field(default_factory=lambda: f"iv_{uuid.uuid4().hex[:10]}")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    state: InterventionState = InterventionState.PENDING

    capability_id: str
    goal: str
    step_id: str
    reason: str

    location: str | None = None
    screen_summary: str | None = None
    screenshot_path: str | None = None

    # How the operator reaches the live session. This is the handoff, concretely.
    session_endpoint: str | None = None

    claimed_by: str | None = None
    claimed_at: str | None = None
    resolved_at: str | None = None
    resume_instruction: str | None = None
    human_actions: list[HumanAction] = Field(default_factory=list)


class SessionLease(BaseModel):
    holder: Controller = Controller.AUTOMATION
    holder_id: str | None = None
    since: float = Field(default_factory=time.time)
    expires_at: float | None = None
    intervention_id: str | None = None

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at


class EscalationBroker:
    """Owns the lease and the intervention queue for one session.

    In production this is a service with a queue and a real console. The seam is the
    same: raise, claim, release. Making it in-process here keeps the demo honest about
    what it is, without pretending the control-transfer model is simpler than it is.
    """

    def __init__(self, evidence, operator_lease_seconds: int = 900, session_endpoint: str | None = None):
        self.evidence = evidence
        self.lease = SessionLease()
        self.requests: dict[str, InterventionRequest] = {}
        self.operator_lease_seconds = operator_lease_seconds
        self.session_endpoint = session_endpoint

    # --- raise ---------------------------------------------------------------

    def raise_intervention(
        self,
        capability_id: str,
        goal: str,
        step_id: str,
        reason: str,
        observation: Any | None = None,
        surface: Any | None = None,
    ) -> InterventionRequest:
        request = InterventionRequest(
            capability_id=capability_id, goal=goal, step_id=step_id, reason=reason,
            session_endpoint=self.session_endpoint,
        )
        if observation is not None:
            request.location = observation.location
            request.screen_summary = observation.summarise(limit=40)
        if surface is not None:
            try:
                shot = self.evidence.path(f"intervention_{request.id}.png")
                surface.observe(screenshot=True, screenshot_path=shot)
                request.screenshot_path = shot
            except Exception:
                pass

        self.requests[request.id] = request
        # Automation stops holding the session the moment it asks for help. Leaving the
        # lease with automation while a human is expected to act is how two actors end
        # up typing into the same form.
        self.lease = SessionLease(
            holder=Controller.NOBODY, since=time.time(), intervention_id=request.id
        )
        self.evidence.log(
            "escalation.raised",
            {"id": request.id, "capability": capability_id, "step": step_id, "reason": reason},
        )
        self._persist(request)
        return request

    # --- claim / release -----------------------------------------------------

    def claim(self, intervention_id: str, operator_id: str) -> InterventionRequest:
        request = self._get(intervention_id)
        if request.state not in {InterventionState.PENDING, InterventionState.ABANDONED}:
            raise RuntimeError(f"intervention {intervention_id} is {request.state.value}")
        if self.lease.holder is Controller.OPERATOR and not self.lease.is_expired():
            raise RuntimeError(f"session already held by operator {self.lease.holder_id}")

        request.state = InterventionState.CLAIMED
        request.claimed_by = operator_id
        request.claimed_at = datetime.now(timezone.utc).isoformat()
        self.lease = SessionLease(
            holder=Controller.OPERATOR, holder_id=operator_id, since=time.time(),
            expires_at=time.time() + self.operator_lease_seconds,
            intervention_id=intervention_id,
        )
        self.evidence.log("escalation.claimed", {"id": intervention_id, "operator": operator_id})
        self._persist(request)
        return request

    def record_human_action(self, intervention_id: str, description: str, location: str | None = None) -> None:
        """Audit what the human did on the session. Required in a regulated setting, and
        also the raw material for turning a manual fix into a recorded step later."""
        request = self._get(intervention_id)
        request.human_actions.append(
            HumanAction(
                at=datetime.now(timezone.utc).isoformat(),
                description=description, location=location,
            )
        )
        self.evidence.log(
            "escalation.human_action", {"id": intervention_id, "description": description}
        )
        self._persist(request)

    def release(self, intervention_id: str, resume_instruction: str = "resume") -> InterventionRequest:
        request = self._get(intervention_id)
        request.state = InterventionState.RESOLVED
        request.resolved_at = datetime.now(timezone.utc).isoformat()
        request.resume_instruction = resume_instruction
        self.lease = SessionLease(holder=Controller.AUTOMATION, since=time.time())
        self.evidence.log(
            "escalation.released",
            {"id": intervention_id, "instruction": resume_instruction,
             "human_actions": len(request.human_actions)},
        )
        self._persist(request)
        return request

    def reclaim_if_expired(self) -> bool:
        """An abandoned session returns to the pool rather than being stuck forever."""
        if self.lease.holder is Controller.OPERATOR and self.lease.is_expired():
            intervention_id = self.lease.intervention_id
            if intervention_id and intervention_id in self.requests:
                self.requests[intervention_id].state = InterventionState.ABANDONED
                self._persist(self.requests[intervention_id])
            self.lease = SessionLease(holder=Controller.NOBODY, since=time.time())
            self.evidence.log("escalation.lease_expired", {"id": intervention_id})
            return True
        return False

    # --- introspection -------------------------------------------------------

    def who_is_in_control(self) -> Controller:
        self.reclaim_if_expired()
        return self.lease.holder

    def automation_may_act(self) -> bool:
        return self.who_is_in_control() is Controller.AUTOMATION

    def pending(self) -> list[InterventionRequest]:
        return [r for r in self.requests.values() if r.state is InterventionState.PENDING]

    def _get(self, intervention_id: str) -> InterventionRequest:
        if intervention_id not in self.requests:
            raise KeyError(f"unknown intervention {intervention_id}")
        return self.requests[intervention_id]

    def _persist(self, request: InterventionRequest) -> None:
        """Interventions outlive the process; an operator console reads them from here."""
        self.evidence.write_json(f"intervention_{request.id}.json", request.model_dump())
