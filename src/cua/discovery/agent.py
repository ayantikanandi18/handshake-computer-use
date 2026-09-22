"""
The discovery agent: an LLM drives the surface until the goal is met.

This is the only place in the system where a model is in the decision loop. Everything
downstream - replay, policy checks, escalation - runs without it. That separation is the
whole architecture: the model is an expensive, non-deterministic way to *learn* a flow,
and a terrible way to *repeat* one.

Three deliberate constraints on the loop:

* One action per turn, chosen from a closed grammar. No free-form code, no selector
  strings invented by the model. The model picks a control by the ref it was shown,
  which means it can only act on things that actually exist on screen.
* Every action is checked against policy before it executes, not after. The model is
  untrusted input; the allowlist is the boundary.
* The model never sees raw secrets. Credentials are referenced by name and substituted
  at the surface, so they cannot be echoed into a transcript or an artifact.
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from cua.discovery.llm import LLMClient
from cua.evidence.recorder import EvidenceRecorder
from cua.safety.policy import PolicyDecision, PolicyEngine
from cua.surface.base import ActionRequest, Observation, Surface

MAX_STEPS_DEFAULT = 25


class AgentAction(BaseModel):
    """The closed grammar. Anything outside this cannot be expressed, let alone run."""

    reasoning: str = Field(description="One sentence: why this action, now.")
    action: Literal["navigate", "click", "fill", "select", "extract", "done", "give_up"]
    ref: str | None = Field(default=None, description="Control ref from the CONTROLS list.")
    value: str | None = Field(default=None, description="Text to type, or option to select.")
    url: str | None = Field(default=None, description="For navigate.")
    # Populated on `done` so the recorder knows what the run actually produced.
    extracted: dict[str, str] | None = Field(
        default=None, description="On done: the values the goal asked for."
    )
    reason_stuck: str | None = Field(default=None, description="On give_up: what blocked it.")

    @field_validator("ref", mode="before")
    @classmethod
    def _normalise_ref(cls, value):
        """Accept the ref however the model echoes it back.

        Controls are shown to the model as "[e4]", and models routinely return the
        bracketed form. Rejecting that would fail a run over punctuation, so the
        brackets are stripped at the boundary instead. Cheap tolerance at the edge,
        strict matching everywhere inside.
        """
        if isinstance(value, str):
            cleaned = value.strip().strip("[]").strip()
            return cleaned or None
        return value


class DiscoveryStep(BaseModel):
    index: int
    observation_summary: str
    action: AgentAction
    policy: str
    result_ok: bool
    result_detail: str
    element_snapshot: dict | None = None


class DiscoveryRun(BaseModel):
    goal: str
    entry_url: str
    model: str
    success: bool
    steps: list[DiscoveryStep] = Field(default_factory=list)
    extracted: dict[str, str] = Field(default_factory=dict)
    stopped_because: str = ""
    escalation_id: str | None = None


SYSTEM_PROMPT = """You operate a bank back-office web application, one action at a time.

You will be shown the controls currently on screen. Each has a ref like [e4]. You may
only act on controls that appear in that list, by their ref.

Rules:
- Return exactly one action per turn, as JSON.
- Prefer the control whose label or name matches what you need.
- This app uses frames. A control's frame is shown; you do not need to switch frames
  yourself, just use the ref.
- To finish, return action "done" and put the values the goal asked for in "extracted".
- If you are blocked and cannot progress, return "give_up" with reason_stuck.
- Never invent a ref. If what you need is not on screen, navigate or click to reach it.

Actions:
  navigate  - url required
  click     - ref required
  fill      - ref and value required
  select    - ref and value required (value is the option text or value)
  extract   - ref required; read a value off screen
  done      - goal achieved; set extracted
  give_up   - blocked; set reason_stuck
"""


class DiscoveryAgent:
    def __init__(
        self,
        surface: Surface,
        llm: LLMClient,
        policy: PolicyEngine,
        evidence: EvidenceRecorder,
        max_steps: int = MAX_STEPS_DEFAULT,
    ):
        self.surface = surface
        self.llm = llm
        self.policy = policy
        self.evidence = evidence
        self.max_steps = max_steps

    def run(self, goal: str, entry_url: str, secrets: dict[str, str] | None = None) -> DiscoveryRun:
        secrets = secrets or {}
        run = DiscoveryRun(goal=goal, entry_url=entry_url, model=self.llm.model, success=False)

        gate = self.policy.check_navigation(entry_url)
        if not gate.allowed:
            run.stopped_because = f"entry url blocked by policy: {gate.reason}"
            self.evidence.log("policy.block", {"url": entry_url, "reason": gate.reason})
            return run

        self.surface.act(ActionRequest(kind="navigate", url=entry_url))

        history: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        for index in range(1, self.max_steps + 1):
            observation = self.surface.observe(
                screenshot=True, screenshot_path=self.evidence.path(f"step{index:02d}.png")
            )
            summary = observation.summarise()
            self.evidence.log(
                "observe",
                {"step": index, "location": observation.location,
                 "controls": len(observation.elements)},
            )

            # A compact record of what has already been done. Without this the agent has
            # no working memory and re-does its last action forever: the first version of
            # this loop filled the same field twelve times because a fill barely changes
            # the screen, so every turn looked like the first one. The full transcript is
            # the wrong fix - it degrades instruction-following on a small model - so the
            # scratchpad carries just the actions and their outcomes.
            # Refs are renumbered on every observation, so "clicked e6" is worse than
            # useless one screen later: it names a different control. Recording what the
            # control *was* keeps the history meaningful across screens.
            def _what(step) -> str:
                snap = step.element_snapshot or {}
                return str(
                    snap.get("name") or snap.get("label")
                    or step.action.ref or step.action.url or ""
                )[:40]

            done_so_far = "\n".join(
                f"  {s.index}. {s.action.action} \"{_what(s)}\""
                f"{(' = ' + s.action.value) if s.action.value else ''}"
                f" -> {'ok' if s.result_ok else 'FAILED: ' + s.result_detail[:50]}"
                for s in run.steps
            ) or "  (nothing yet)"

            prompt = (
                f"GOAL: {goal}\n\n"
                f"ACTIONS YOU HAVE ALREADY TAKEN:\n{done_so_far}\n\n"
                f"CURRENT SCREEN:\n{summary}\n\n"
                "Choose the single next action. Do not repeat an action that already "
                "succeeded; move on to the next part of the goal."
            )
            messages = history + [{"role": "user", "content": prompt}]
            action = self.llm.decide(messages, AgentAction)
            self.evidence.log(
                "decide",
                {"step": index, "action": action.action, "ref": action.ref,
                 "reasoning": action.reasoning[:300]},
            )

            if action.action == "done":
                run.extracted = action.extracted or {}
                run.success = True
                run.stopped_because = "agent reported goal achieved"
                run.steps.append(
                    DiscoveryStep(
                        index=index, observation_summary=summary, action=action,
                        policy="n/a", result_ok=True, result_detail="done",
                    )
                )
                break

            if action.action == "give_up":
                run.stopped_because = f"agent gave up: {action.reason_stuck}"
                run.steps.append(
                    DiscoveryStep(
                        index=index, observation_summary=summary, action=action,
                        policy="n/a", result_ok=False,
                        result_detail=action.reason_stuck or "stuck",
                    )
                )
                break

            decision = self._authorise(action, observation)
            if not decision.allowed:
                # A blocked action is not a crash: it is information. Feed it back so the
                # model can choose a permitted route instead of repeating itself.
                self.evidence.log(
                    "policy.block",
                    {"step": index, "action": action.action, "reason": decision.reason},
                )
                run.steps.append(
                    DiscoveryStep(
                        index=index, observation_summary=summary, action=action,
                        policy=f"blocked: {decision.reason}", result_ok=False,
                        result_detail=decision.reason,
                    )
                )
                history.append({"role": "user", "content": prompt})
                history.append(
                    {
                        "role": "assistant",
                        "content": f'{{"action":"{action.action}"}}',
                    }
                )
                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"That action was blocked by policy: {decision.reason}. "
                            "Choose a different, permitted action."
                        ),
                    }
                )
                continue

            element = self._element_for(action.ref, observation)
            value = self._resolve_value(action.value, secrets)
            result = self.surface.act(
                ActionRequest(
                    kind=action.action if action.action != "extract" else "wait",
                    ref=action.ref, value=value, url=action.url,
                )
            )
            if action.action == "extract" and element is not None:
                run.extracted[element.label or element.name or action.ref] = element.text

            self.evidence.log(
                "act",
                {"step": index, "action": action.action, "ok": result.ok,
                 "detail": result.detail[:300]},
            )
            run.steps.append(
                DiscoveryStep(
                    index=index, observation_summary=summary, action=action,
                    policy="allowed", result_ok=result.ok, result_detail=result.detail,
                    element_snapshot=element.model_dump() if element else None,
                )
            )

            # Keep the conversation short. The screen is re-sent every turn anyway, and a
            # long transcript on a small model degrades instruction-following fast.
            history = history[:1]
            time.sleep(0.2)
        else:
            run.stopped_because = f"hit max steps ({self.max_steps})"

        self.evidence.log("run.end", {"success": run.success, "why": run.stopped_because})
        return run

    # --- helpers -------------------------------------------------------------

    def _authorise(self, action: AgentAction, observation: Observation) -> PolicyDecision:
        if action.action == "navigate":
            return self.policy.check_navigation(action.url or "")
        element = self._element_for(action.ref, observation)
        return self.policy.check_action(
            action_kind=action.action,
            element_role=element.role if element else None,
            element_name=(element.name or element.label or "") if element else "",
            location=observation.location,
        )

    @staticmethod
    def _element_for(ref: str | None, observation: Observation):
        if not ref:
            return None
        for el in observation.elements:
            if el.ref == ref:
                return el
        return None

    @staticmethod
    def _resolve_value(value: str | None, secrets: dict[str, str]) -> str | None:
        """Substitute secret references at the boundary.

        The model asks for "{{secret:operator_password}}"; the real value is put in by
        the surface. The model never sees it, so it cannot end up in a transcript, an
        artifact, or a log.
        """
        if not value:
            return value
        if value.startswith("{{secret:") and value.endswith("}}"):
            key = value[len("{{secret:") : -2].strip()
            return secrets.get(key, "")
        return value
