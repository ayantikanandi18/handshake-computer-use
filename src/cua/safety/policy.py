"""
Policy: the boundary between what the model proposes and what actually happens.

The model is untrusted input. It is not malicious, but it is a text generator that will
occasionally decide the right move is to click "Post Transactions" or wander onto a
domain nobody approved. So policy is enforced at the point of action, on every action,
in both discovery and replay - not as a prompt instruction, which is a suggestion.

The risk model is deliberately coarse and conservative:

  READ          navigation, searching, reading           -> allowed
  WRITE         creates or changes a record               -> allowed only if the policy
                                                             says so for this capability
  IRREVERSIBLE  money movement, deletion, notification    -> never auto-approved

The default for an unrecognised control is WRITE, not READ. On a bank back-office screen
the cost of wrongly assuming a button is harmless is much higher than the cost of
stopping to ask, and "fail closed" is the only defensible default in this domain.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from cua.artifact.schema import RiskClass


class PolicyDecision(BaseModel):
    allowed: bool
    risk: RiskClass = RiskClass.READ
    reason: str = ""
    requires_confirmation: bool = False


# Names that indicate an action a bank would not want automated without a human.
IRREVERSIBLE_PATTERNS = [
    r"\btransfer\b", r"\bpost\b", r"\bdisburse\b", r"\bwire\b", r"\bclose\s+account\b",
    r"\bdelete\b", r"\bremove\b", r"\bvoid\b", r"\breverse\b", r"\bcharge\s*off\b",
    r"\bsend\b", r"\bnotify\b", r"\bapprove\b", r"\bpay\b",
]

WRITE_PATTERNS = [
    r"\bsubmit\b", r"\bsave\b", r"\bopen\s+account\b", r"\bcreate\b", r"\bupdate\b",
    r"\badd\b", r"\bpost\b", r"\bapply\b", r"\bconfirm\b",
]

READ_PATTERNS = [
    r"\bsearch\b", r"\bview\b", r"\bfind\b", r"\blook\s*up\b", r"\bback\b", r"\bhome\b",
    r"\bcancel\b", r"\bsign\s*on\b", r"\blogin\b", r"\backnowledge\b", r"\bnext\b",
]


class PolicyConfig(BaseModel):
    """Explicit, configurable allowlist. Nothing outside it is permitted."""

    allowed_origins: list[str] = Field(default_factory=list)
    allowed_path_prefixes: list[str] = Field(default_factory=lambda: ["/"])
    denied_path_patterns: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(
        default_factory=lambda: ["navigate", "click", "fill", "select", "extract", "press"]
    )
    # The highest risk the run may take on its own. Anything above escalates.
    max_autonomous_risk: RiskClass = RiskClass.READ
    # Risky steps can be permitted for a specific, reviewed capability.
    confirmed_write_capabilities: list[str] = Field(default_factory=list)


class PolicyEngine:
    def __init__(self, config: PolicyConfig, capability_id: str | None = None):
        self.config = config
        self.capability_id = capability_id

    # --- navigation ----------------------------------------------------------

    def check_navigation(self, url: str) -> PolicyDecision:
        if not url:
            return PolicyDecision(allowed=False, reason="empty url")
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if self.config.allowed_origins and origin not in self.config.allowed_origins:
            return PolicyDecision(
                allowed=False,
                reason=f"origin {origin} is not in the allowlist",
            )
        path = parsed.path or "/"
        for pattern in self.config.denied_path_patterns:
            if re.search(pattern, path, re.IGNORECASE):
                return PolicyDecision(allowed=False, reason=f"path {path} matches deny rule {pattern}")
        if self.config.allowed_path_prefixes and not any(
            path.startswith(p) for p in self.config.allowed_path_prefixes
        ):
            return PolicyDecision(allowed=False, reason=f"path {path} is outside the allowed prefixes")
        return PolicyDecision(allowed=True, risk=RiskClass.READ, reason="navigation permitted")

    # --- actions -------------------------------------------------------------

    @staticmethod
    def normalise_control_name(raw: str) -> str:
        """Make a legacy control name readable before classifying it.

        Back-office apps name controls btnSearch, cmdPostTxn, lnkViewDetail. Matching
        \\bsearch\\b against "btnSearch" fails, because there is no word boundary inside
        camelCase - which in practice meant the policy engine classified an ordinary
        search button as an unrecognised write and blocked a read-only flow. Splitting
        camelCase and dropping the hungarian prefix recovers the human-readable intent
        without loosening any rule.
        """
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", raw or "")
        text = re.sub(r"[_\-]+", " ", text)
        text = re.sub(
            r"^\s*(btn|cmd|lnk|txt|ddl|chk|rad|img|lbl|grd|gv)\s+", " ", text, flags=re.IGNORECASE
        )
        return text.strip().lower()

    def classify(self, element_role: str | None, element_name: str) -> RiskClass:
        """Classify by what the control says it does.

        Reading intent off the label is imperfect, which is exactly why the default is
        WRITE rather than READ: an unclassifiable control is treated as dangerous until
        a human says otherwise.
        """
        name = self.normalise_control_name(element_name)
        for pattern in IRREVERSIBLE_PATTERNS:
            if re.search(pattern, name):
                return RiskClass.IRREVERSIBLE
        for pattern in READ_PATTERNS:
            if re.search(pattern, name):
                return RiskClass.READ
        for pattern in WRITE_PATTERNS:
            if re.search(pattern, name):
                return RiskClass.WRITE
        if element_role in {"link", "textbox", "combobox", "searchbox"}:
            return RiskClass.READ
        return RiskClass.WRITE  # fail closed

    def check_action(
        self,
        action_kind: str,
        element_role: str | None,
        element_name: str,
        location: str = "",
    ) -> PolicyDecision:
        if action_kind not in self.config.allowed_actions:
            return PolicyDecision(allowed=False, reason=f"action '{action_kind}' is not permitted")

        # Typing and reading do not themselves change state; the submit that follows does.
        if action_kind in {"fill", "select", "extract", "press"}:
            return PolicyDecision(allowed=True, risk=RiskClass.READ, reason="non-committing action")

        risk = self.classify(element_role, element_name)
        order = list(RiskClass)
        ceiling = self.config.max_autonomous_risk

        if risk is RiskClass.IRREVERSIBLE:
            return PolicyDecision(
                allowed=False, risk=risk, requires_confirmation=True,
                reason=(
                    f"'{element_name}' looks irreversible; irreversible actions are never "
                    "taken autonomously and must go to a human"
                ),
            )

        if order.index(risk) > order.index(ceiling):
            if self.capability_id and self.capability_id in self.config.confirmed_write_capabilities:
                return PolicyDecision(
                    allowed=True, risk=risk,
                    reason=f"write permitted: capability '{self.capability_id}' is approved",
                )
            return PolicyDecision(
                allowed=False, risk=risk, requires_confirmation=True,
                reason=(
                    f"'{element_name}' is a {risk.value} action and this run is limited to "
                    f"{ceiling.value}; needs approval or an approved capability"
                ),
            )

        return PolicyDecision(allowed=True, risk=risk, reason=f"{risk.value} action permitted")


def default_policy_for_demo(origin: str = "http://127.0.0.1:8099") -> PolicyConfig:
    """The policy used by the demo runs. Narrow on purpose."""
    return PolicyConfig(
        allowed_origins=[origin],
        allowed_path_prefixes=["/"],
        # The fault switchboard is a test affordance, not something automation may touch.
        denied_path_patterns=[r"^/admin/"],
        max_autonomous_risk=RiskClass.READ,
    )
