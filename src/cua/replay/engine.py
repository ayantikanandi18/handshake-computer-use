"""
Deterministic replay: the production execution path.

No model is involved here. Given an artifact and a set of parameters, this runs the
recorded flow, checks what it expects to see at every step, handles the runtime
conditions the artifact declares, and returns a typed result.

The ordering inside each step is the part that carries the weight:

  1. classify the screen first  - before acting, check whether the app is showing a
                                  declared business outcome or a recoverable condition.
                                  A flow that types a member ID into a session-expired
                                  login page is a flow that did not look first.
  2. check policy               - the recorded flow is data, and data can be wrong or
                                  tampered with. It gets authorised on every invocation,
                                  not trusted because it was approved at record time.
  3. resolve the target         - layered strategies, exactly one match required.
  4. act
  5. assert the expectation     - a step that ran is not a step that worked.

Recovery is bounded and declared. Replay may dismiss a known interstitial, wait out a
slow load, or re-authenticate - because those are conditions the artifact says can
happen and says what to do about. It may not improvise. Anything undeclared stops the
run, which is the conservative choice and the right one when the subject is a core
banking system.
"""

from __future__ import annotations

import re
import time
from typing import Any

from cua.artifact.schema import (
    ActionType, BusinessOutcome, Capability, RecoverableCondition, RecoveryAction,
    RiskClass, StateAssertion, Step,
)
from cua.evidence.recorder import EvidenceRecorder
from cua.replay.outcomes import (
    FailureKind, RecoveryRecord, ReplayResult, ResultStatus, StepTrace,
)
from cua.safety.policy import PolicyEngine
from cua.surface.base import ActionRequest, Observation, Surface


class ReplayEngine:
    def __init__(
        self,
        surface: Surface,
        policy: PolicyEngine,
        evidence: EvidenceRecorder,
        escalation=None,
    ):
        self.surface = surface
        self.policy = policy
        self.evidence = evidence
        self.escalation = escalation
        # Per-instance: a class attribute here would share credentials between runs.
        self._secrets_cache: dict[str, str] = {}

    # --- entry point ---------------------------------------------------------

    def run(
        self,
        capability: Capability,
        params: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        secrets: dict[str, str] | None = None,
    ) -> ReplayResult:
        started = time.time()
        params = params or {}
        secrets = secrets or {}
        flow = capability.resolved_for(tenant_id)

        result = ReplayResult(
            status=ResultStatus.SUCCESS,
            capability_id=flow.id,
            capability_version=flow.version,
            tenant_id=tenant_id,
            evidence_dir=str(self.evidence.dir),
        )

        problem = self._validate_params(flow, params)
        if problem:
            return self._fail(result, FailureKind.ASSERTION_FAILED, "<params>", problem, "", started)

        self.evidence.log(
            "replay.start",
            {"capability": flow.id, "version": flow.version, "tenant": tenant_id,
             "params": {k: ("[REDACTED]" if self._is_sensitive(flow, k) else v)
                        for k, v in params.items()}},
        )

        # Start where the artifact says the flow starts. Discovery navigated here before
        # its loop began, so the recorded steps assume it; without this replay runs the
        # first step against a blank page. The entry point goes through the same policy
        # check as any other navigation - it is data from an artifact, not a trusted
        # constant.
        entry = flow.surface.entry_url
        if entry:
            nav = self.policy.check_navigation(entry)
            if not nav.allowed:
                return self._fail(
                    result, FailureKind.POLICY_BLOCKED, "<entry>", "permitted entry url",
                    nav.reason, started,
                )
            opened = self.surface.act(ActionRequest(kind="navigate", url=entry))
            if not opened.ok:
                return self._fail(
                    result, FailureKind.SURFACE_ERROR, "<entry>",
                    f"navigate to {entry}", opened.detail, started,
                )

        for step in flow.steps:
            step_started = time.time()
            observation = self.surface.observe()

            # 0. Is the application itself broken? A 5xx is not a business outcome and
            # not something an operator taking over the session can fix by clicking, so
            # it stops the run immediately rather than escalating or hunting for
            # controls on an error page.
            if observation.http_status and observation.http_status >= 500:
                self.evidence.capture_failure(
                    self.surface, step.id, "a working application screen",
                    f"HTTP {observation.http_status}",
                )
                return self._fail(
                    result, FailureKind.SURFACE_ERROR, step.id,
                    "a working application screen",
                    f"application returned HTTP {observation.http_status}", started,
                )

            # 1. Is the app telling us something before we act?
            outcome = self._match_business_outcome(flow, observation)
            if outcome:
                self.evidence.log("replay.business_outcome",
                                  {"code": outcome.code, "at_step": step.id})
                result.status = ResultStatus.BUSINESS_OUTCOME
                result.outcome_code = outcome.code
                result.outcome_detail = outcome.description
                result.duration_ms = int((time.time() - started) * 1000)
                return result

            observation, recovered = self._attempt_recovery(flow, observation, step, result)
            if recovered is False:
                return self._fail(
                    result, FailureKind.UNRECOVERABLE_CONDITION, step.id,
                    "a recoverable condition did not clear", observation.text[:300], started,
                )

            # 2. Authorise on every invocation.
            #
            # Two opinions about this step's risk exist: the one recorded in the
            # artifact (reviewed by a human before the capability was approved) and the
            # one the policy engine infers from the control's name at runtime. They can
            # disagree - "View" reads as harmless, but a reviewer may have marked that
            # step WRITE because of what the screen behind it does. Neither is
            # authoritative, so the conservative one wins: an action is gated at the
            # highest risk anyone assigned it.
            decision = self.policy.check_action(
                action_kind=step.action.value,
                element_role=None,
                element_name=(step.target.description if step.target else step.description),
                location=observation.location,
            )
            order = list(RiskClass)
            effective_risk = max(step.risk, decision.risk, key=lambda r: order.index(r))
            if order.index(effective_risk) > order.index(self.policy.config.max_autonomous_risk):
                decision.allowed = False
                decision.risk = effective_risk
                decision.reason = (
                    f"step is classified {effective_risk.value} "
                    f"(recorded {step.risk.value}, inferred {decision.risk.value}); "
                    f"this run is limited to {self.policy.config.max_autonomous_risk.value}"
                )

            if effective_risk is not RiskClass.READ and not decision.allowed:
                if self.escalation:
                    return self._escalate(
                        result, flow, step, observation,
                        f"policy requires human approval: {decision.reason}", started,
                    )
                return self._fail(
                    result, FailureKind.POLICY_BLOCKED, step.id, "permitted action",
                    decision.reason, started,
                )
            if step.action is ActionType.NAVIGATE:
                nav = self.policy.check_navigation(step.url or "")
                if not nav.allowed:
                    return self._fail(
                        result, FailureKind.POLICY_BLOCKED, step.id, "permitted url",
                        nav.reason, started,
                    )

            # 3-5. resolve, act, assert
            trace = self._execute(flow, step, params, secrets, result)
            trace.duration_ms = int((time.time() - step_started) * 1000)
            result.trace.append(trace)

            if not trace.ok:
                if step.optional:
                    self.evidence.log("replay.step_skipped", {"step": step.id, "why": trace.detail})
                    continue
                kind = self._failure_kind(trace)
                if self.escalation and kind in {
                    FailureKind.TARGET_NOT_FOUND, FailureKind.AMBIGUOUS_TARGET,
                }:
                    observation = self.surface.observe()
                    return self._escalate(result, flow, step, observation, trace.detail, started)
                self.evidence.capture_failure(
                    self.surface, step.id, trace.expected or "", trace.observed or trace.detail
                )
                return self._fail(
                    result, kind, step.id, trace.expected or "", trace.observed or trace.detail,
                    started,
                )

        # checkpoint: did the flow actually achieve the goal?
        final = self.surface.observe()
        outcome = self._match_business_outcome(flow, final)
        if outcome:
            result.status = ResultStatus.BUSINESS_OUTCOME
            result.outcome_code = outcome.code
            result.outcome_detail = outcome.description
            result.duration_ms = int((time.time() - started) * 1000)
            return result

        if not self._assert_state(flow.checkpoint, final):
            self.evidence.capture_failure(
                self.surface, "<checkpoint>", flow.checkpoint.value, final.text[:400]
            )
            return self._fail(
                result, FailureKind.CHECKPOINT_FAILED, "<checkpoint>",
                flow.checkpoint.value, final.text[:300], started,
            )

        outputs, missing = self._extract_outputs(flow, final)
        if missing:
            return self._fail(
                result, FailureKind.OUTPUT_EXTRACTION_FAILED, "<outputs>",
                f"outputs {missing}", final.text[:300], started,
            )
        result.outputs = outputs
        result.duration_ms = int((time.time() - started) * 1000)
        self.evidence.log("replay.success", {"outputs_keys": list(outputs)})
        return result

    # --- step execution ------------------------------------------------------

    def _execute(self, flow, step: Step, params, secrets, result) -> StepTrace:
        trace = StepTrace(step_id=step.id, action=step.action.value,
                          description=step.description, ok=False)

        if step.action is ActionType.NAVIGATE:
            res = self.surface.act(
                ActionRequest(kind="navigate", url=step.url, timeout_ms=step.timeout_ms)
            )
            trace.ok, trace.detail = res.ok, res.detail
        elif step.action is ActionType.ASSERT:
            observation = self.surface.observe()
            ok = self._assert_state(step.expect, observation) if step.expect else True
            trace.ok = ok
            trace.expected = step.expect.value if step.expect else None
            trace.observed = None if ok else observation.text[:300]
            trace.detail = "assertion held" if ok else "assertion failed"
            return trace
        else:
            locator, tier = self.surface.resolve(step.target, None)
            trace.matched_tier = tier
            if locator is None:
                trace.detail = f"could not resolve target: {step.target.description}"
                trace.expected = step.target.description
                return trace
            if tier and step.target.strategies and tier != step.target.strategies[0].tier.value:
                # Drift signal: we found it, but not the way we recorded it.
                self.evidence.log(
                    "replay.tier_fallback",
                    {"step": step.id, "recorded": step.target.strategies[0].tier.value,
                     "matched": tier},
                )
            value = self._bind_value(step.value, params, secrets)
            try:
                if step.action is ActionType.CLICK:
                    locator.click(timeout=step.timeout_ms)
                elif step.action is ActionType.FILL:
                    locator.fill(value or "", timeout=step.timeout_ms)
                elif step.action is ActionType.SELECT:
                    locator.select_option(value or "", timeout=step.timeout_ms)
                elif step.action is ActionType.PRESS:
                    locator.press(value or "Enter", timeout=step.timeout_ms)
                elif step.action in {ActionType.WAIT_FOR, ActionType.EXTRACT}:
                    locator.wait_for(state="visible", timeout=step.timeout_ms)
                trace.ok = True
                trace.detail = f"{step.action.value} ok (tier={tier})"
            except Exception as exc:
                trace.detail = f"{type(exc).__name__}: {str(exc)[:200]}"
                return trace

        if trace.ok and step.expect:
            observation = self.surface.observe()
            if not self._assert_state(step.expect, observation):
                trace.ok = False
                trace.expected = step.expect.value
                trace.observed = observation.text[:300]
                trace.detail = "post-step assertion failed"
        return trace

    # --- conditions ----------------------------------------------------------

    def _match_business_outcome(self, flow: Capability, obs: Observation) -> BusinessOutcome | None:
        for outcome in flow.business_outcomes:
            if self._assert_state(outcome.detect, obs):
                return outcome
        return None

    def _attempt_recovery(self, flow, obs: Observation, step, result):
        """Handle declared runtime conditions. Returns (observation, recovered_or_None)."""
        for condition in flow.recoverables:
            if not self._assert_state(condition.detect, obs):
                continue
            for attempt in range(1, condition.max_attempts + 1):
                self.evidence.log(
                    "replay.recovery",
                    {"condition": condition.code, "action": condition.action.value,
                     "attempt": attempt, "at_step": step.id},
                )
                ok = self._apply_recovery(condition, flow)
                obs = self.surface.observe()
                cleared = not self._assert_state(condition.detect, obs)
                if ok and cleared:
                    result.recoveries.append(
                        RecoveryRecord(condition_code=condition.code,
                                       action=condition.action.value, attempts=attempt,
                                       succeeded=True, at_step=step.id)
                    )
                    return obs, True
            result.recoveries.append(
                RecoveryRecord(condition_code=condition.code, action=condition.action.value,
                               attempts=condition.max_attempts, succeeded=False, at_step=step.id)
            )
            return obs, False
        return obs, None

    def _apply_recovery(self, condition: RecoverableCondition, flow: Capability) -> bool:
        if condition.action is RecoveryAction.WAIT_RETRY:
            self.surface.act(ActionRequest(kind="wait", timeout_ms=3000))
            return True
        if condition.action is RecoveryAction.DISMISS and condition.target:
            locator, _ = self.surface.resolve(condition.target, None)
            if locator is None:
                return False
            try:
                locator.click(timeout=5000)
                return True
            except Exception:
                return False
        if condition.action is RecoveryAction.REAUTHENTICATE:
            # Re-running the login prefix is deliberate: credentials live in the secrets
            # map, never in the artifact, so this is the only place they can re-enter.
            return self._reauthenticate(flow)
        return False

    def _reauthenticate(self, flow: Capability) -> bool:
        by_id = {s.id: s for s in flow.steps}
        prefix = [by_id[sid] for sid in flow.reauth_step_ids if sid in by_id]
        if not prefix:
            return False
        for step in prefix:
            trace = self._execute(flow, step, {}, self._secrets_cache, ReplayResult(
                status=ResultStatus.SUCCESS, capability_id=flow.id,
                capability_version=flow.version))
            if not trace.ok:
                return False
        return True

    # --- assertions, params, outputs -----------------------------------------

    @staticmethod
    def _assert_state(assertion: StateAssertion | None, obs: Observation) -> bool:
        if assertion is None:
            return True
        haystack = obs.text if assertion.case_sensitive else obs.text.lower()
        needle = assertion.value if assertion.case_sensitive else assertion.value.lower()
        if assertion.kind == "text_present":
            return needle in haystack
        if assertion.kind == "text_absent":
            return needle not in haystack
        if assertion.kind == "url_matches":
            return re.search(assertion.value, obs.location or "") is not None
        if assertion.kind == "http_status":
            return str(obs.http_status) == assertion.value
        if assertion.kind == "element_present":
            return any(
                needle in (el.name or "").lower() or needle in (el.label or "").lower()
                for el in obs.elements
            )
        return False

    @staticmethod
    def _is_sensitive(flow: Capability, name: str) -> bool:
        for param in flow.inputs:
            if param.name == name and param.sensitivity.value in {"pii", "secret"}:
                return True
        return False

    @staticmethod
    def _validate_params(flow: Capability, params: dict) -> str | None:
        for param in flow.inputs:
            if param.required and param.name not in params:
                return f"required input '{param.name}' was not supplied"
            if param.name in params and param.pattern:
                if not re.fullmatch(param.pattern, str(params[param.name])):
                    return f"input '{param.name}' does not match {param.pattern}"
        return None

    def _bind_value(self, value: str | None, params: dict, secrets: dict) -> str | None:
        if not value:
            return value
        text = value.strip()
        if text.startswith("{{secret:") and text.endswith("}}"):
            self._secrets_cache = secrets
            return secrets.get(text[len("{{secret:") : -2].strip(), "")
        if text.startswith("{{") and text.endswith("}}"):
            return str(params.get(text.strip("{} ").strip(), ""))
        return value

    def _extract_outputs(self, flow: Capability, obs: Observation):
        outputs: dict[str, Any] = {}
        missing: list[str] = []
        for spec in flow.outputs:
            locator, _ = self.surface.resolve(spec.source, obs)
            if locator is None:
                missing.append(spec.name)
                continue
            try:
                raw = (locator.inner_text(timeout=5000) or "").strip()
            except Exception:
                missing.append(spec.name)
                continue
            outputs[spec.name] = self._transform(raw, spec)
        return outputs, missing

    @staticmethod
    def _transform(raw: str, spec) -> Any:
        if spec.transform == "strip":
            return raw.strip()
        if spec.transform == "currency_to_number":
            cleaned = re.sub(r"[^0-9.\-]", "", raw)
            try:
                return float(cleaned)
            except ValueError:
                return raw
        if spec.type == "integer":
            try:
                return int(re.sub(r"[^0-9\-]", "", raw))
            except ValueError:
                return raw
        return raw

    # --- terminal states -----------------------------------------------------

    @staticmethod
    def _failure_kind(trace: StepTrace) -> FailureKind:
        detail = (trace.detail or "").lower()
        if "could not resolve" in detail:
            return FailureKind.TARGET_NOT_FOUND
        if "timeout" in detail:
            return FailureKind.TIMEOUT
        if "assertion" in detail:
            return FailureKind.ASSERTION_FAILED
        return FailureKind.SURFACE_ERROR

    def _fail(self, result, kind, step_id, expected, observed, started) -> ReplayResult:
        result.status = ResultStatus.FAILURE
        result.failure_kind = kind
        result.failed_step = step_id
        result.expected = expected
        result.observed = observed
        result.duration_ms = int((time.time() - started) * 1000)
        self.evidence.log(
            "replay.failure",
            {"kind": kind.value, "step": step_id, "expected": expected, "observed": observed[:300]},
        )
        return result

    def _escalate(self, result, flow, step, observation, why, started) -> ReplayResult:
        request = self.escalation.raise_intervention(
            capability_id=flow.id, goal=flow.description, step_id=step.id,
            reason=why, observation=observation, surface=self.surface,
        )
        result.status = ResultStatus.ESCALATED
        result.intervention_id = request.id
        result.failed_step = step.id
        result.observed = why
        result.duration_ms = int((time.time() - started) * 1000)
        return result
