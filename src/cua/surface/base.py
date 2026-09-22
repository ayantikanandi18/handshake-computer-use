"""
The surface abstraction: the seam between "how we perceive and act" and "the recorded flow".

Everything above this line - the artifact, the replay engine, the policy checks, the
escalation logic - is written against these types and never touches Playwright. That is
the whole point. A desktop surface (UIA/AX APIs) or a terminal surface (3270 screen
buffer) implements the same three operations and the rest of the system does not change.

The observation model is deliberately the *operator's* view of a screen: a list of
controls, each with a role, a name, the text near it, and where it is. That is available
from an accessibility tree, from a DOM, from a native automation API, and - with OCR -
from pixels. It is the largest common denominator across every surface in the brief, and
choosing it is what keeps the artifact from silently becoming a pile of CSS selectors.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Element(BaseModel):
    """One control as an operator would perceive it."""

    ref: str = Field(description="Opaque handle the surface uses to act on this element.")
    role: str = Field(description="Accessible role: button, textbox, link, combobox, cell...")
    name: str = Field(default="", description="Accessible name.")
    text: str = Field(default="", description="Visible text content, trimmed.")
    frame: str | None = None
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="x, y, w, h. Present for coordinate-capable surfaces."
    )
    label: str | None = Field(default=None, description="Nearest associated label text.")
    # Best-effort surface-specific detail. Web fills name=/id=/href; a desktop surface
    # would fill AutomationId/ClassName. Nothing above this layer may depend on it.
    attrs: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    visible: bool = True


class Observation(BaseModel):
    """A snapshot of the surface at one moment."""

    location: str = Field(description="URL for web, window title for desktop.")
    title: str = ""
    frames: list[str] = Field(default_factory=list)
    frame_locations: dict[str, str] = Field(
        default_factory=dict, description="Per-frame URL; the real position signal in a frameset."
    )
    elements: list[Element] = Field(default_factory=list)
    text: str = Field(default="", description="Visible text of the surface, for assertions.")
    http_status: int | None = None
    screenshot_path: str | None = None
    aria: str = Field(
        default="", description="ARIA/accessibility tree view; the portable shape of the screen."
    )

    def summarise(self, limit: int = 60) -> str:
        """Compact, token-cheap rendering for the model.

        Sending raw HTML to the model is the thing that makes this approach fail on
        legacy surfaces - the markup is enormous, mostly layout tables, and teaches the
        model to reach for selectors that will not survive. Sending the operator's view
        keeps the prompt small and keeps the model reasoning in terms the artifact can
        actually record.
        """
        lines = [f"LOCATION: {self.location}", f"TITLE: {self.title}"]
        # In a frameset the top-level URL never changes, so it is useless as a position
        # indicator - the agent's first run through this app sat at /app for fourteen
        # steps with no idea which screen it was on. The per-frame locations are the
        # signal that actually tells it where it is.
        if self.frame_locations:
            for frame_name, loc in self.frame_locations.items():
                lines.append(f"FRAME {frame_name}: {loc}")
        elif self.frames:
            lines.append(f"FRAMES: {', '.join(self.frames)}")
        lines.append("CONTROLS:")
        for el in self.elements[:limit]:
            bits = [f"[{el.ref}]", el.role]
            if el.name:
                bits.append(f'name="{el.name[:60]}"')
            if el.label and el.label != el.name:
                bits.append(f'label="{el.label[:40]}"')
            if el.text and el.text != el.name:
                bits.append(f'text="{el.text[:60]}"')
            # Show what is currently typed into an input. An operator can see the
            # contents of a field they just filled; without this the agent cannot
            # observe the effect of its own typing and will fill the same box forever.
            current = el.attrs.get("value")
            if current and el.role in {"textbox", "combobox", "searchbox"}:
                bits.append(f'current="{current[:40]}"')
            if el.frame:
                bits.append(f"frame={el.frame}")
            if not el.enabled:
                bits.append("(disabled)")
            lines.append("  " + " ".join(bits))
        if len(self.elements) > limit:
            lines.append(f"  ... {len(self.elements) - limit} more controls")
        body = self.text.strip()
        if body:
            lines.append("VISIBLE TEXT:")
            lines.append("  " + body[:1200].replace("\n", "\n  "))
        return "\n".join(lines)


class ActionRequest(BaseModel):
    """An action expressed against the observation, not against the underlying tech."""

    kind: str  # navigate | click | fill | select | press | wait
    ref: str | None = None
    value: str | None = None
    url: str | None = None
    timeout_ms: int = 10_000


class ActionResult(BaseModel):
    ok: bool
    detail: str = ""
    raised: str | None = None


@runtime_checkable
class Surface(Protocol):
    """Three operations. Any surface that can do these can host a capability."""

    kind: str

    def observe(self, screenshot: bool = False) -> Observation: ...

    def act(self, request: ActionRequest) -> ActionResult: ...

    def resolve(self, target: Any, observation: Observation) -> tuple[Element | None, str | None]:
        """Find the element a recorded Target refers to.

        Returns (element, tier_that_matched). Returning the tier is not incidental: it is
        the cheapest drift signal the system gets. A capability that used to match on
        role+name and now only matches on dom_path is telling you the app moved under it,
        well before it breaks outright.
        """
        ...

    def close(self) -> None: ...
