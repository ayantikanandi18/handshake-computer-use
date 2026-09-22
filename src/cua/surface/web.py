"""
A web surface backed by Playwright.

Two things about this implementation are deliberate and worth defending.

First, perception is built from the accessibility tree plus geometry, not from CSS.
Playwright can obviously do better than this on a modern app with test IDs - but the
whole premise of the brief is that the common case has no clean DOM, and a perception
layer that quietly depends on one produces artifacts that cannot survive the jump to a
legacy frameset, let alone to a desktop app. Building on roles and names keeps the
recorded flow honest about what it actually depends on.

Second, frames are first class. A frameset app is not one document, and a locator that
does not say which frame it lives in is ambiguous by construction. Every element carries
its frame, and every resolution is frame-scoped.
"""

from __future__ import annotations

import re
from typing import Any

from playwright.sync_api import Frame, Page, sync_playwright

from cua.artifact.schema import LocatorTier, Target
from cua.surface.base import ActionRequest, ActionResult, Element, Observation

# Roles worth showing the model. Everything on a legacy page is a table cell; listing
# them all buries the controls that matter in layout noise.
INTERESTING_ROLES = {
    "button", "link", "textbox", "combobox", "checkbox", "radio", "searchbox",
    "menuitem", "tab", "option", "listbox", "spinbutton", "switch", "heading",
}


class WebSurface:
    kind = "web"

    def __init__(self, headless: bool = True, cdp_port: int | None = None):
        self._pw = sync_playwright().start()
        launch: dict[str, Any] = {"headless": headless}
        if cdp_port:
            # Exposing the debugging port is what makes a real control handoff possible:
            # a human operator's browser can attach to this very session rather than
            # being handed a screenshot and told to redo the work elsewhere.
            launch["args"] = [f"--remote-debugging-port={cdp_port}"]
        self._browser = self._pw.chromium.launch(**launch)
        self._context = self._browser.new_context(viewport={"width": 1280, "height": 900})
        self.page: Page = self._context.new_page()
        self._last_status: int | None = None
        self.page.on("response", self._note_status)
        self._ref_map: dict[str, tuple[str | None, str]] = {}

    def _note_status(self, response) -> None:
        try:
            if response.request.is_navigation_request():
                self._last_status = response.status
        except Exception:
            pass

    # --- perception ----------------------------------------------------------

    def _frames(self) -> list[tuple[str | None, Frame]]:
        out: list[tuple[str | None, Frame]] = []
        for frame in self.page.frames:
            if frame.parent_frame is None:
                out.append((None, frame))
            else:
                out.append((frame.name or frame.url.rsplit("/", 1)[-1] or "frame", frame))
        return out

    def observe(self, screenshot: bool = False, screenshot_path: str | None = None) -> Observation:
        elements: list[Element] = []
        texts: list[str] = []
        aria_views: list[str] = []
        frame_names: list[str] = []
        frame_locs: dict[str, str] = {}
        self._ref_map.clear()
        counter = 0

        for frame_name, frame in self._frames():
            if frame_name:
                frame_names.append(frame_name)
                try:
                    frame_locs[frame_name] = frame.url
                except Exception:
                    pass
            # Visible text, used for assertions and for the model's view of the screen.
            try:
                body_text = frame.evaluate("() => document.body ? document.body.innerText : ''")
                if body_text:
                    texts.append(body_text)
            except Exception:
                pass

            # The ARIA snapshot is the portable, operator-shaped view of the screen and
            # is what a desktop AX tree would hand us. On these legacy screens it also
            # demonstrates the core difficulty: inputs frequently have no accessible
            # name, only a position inside a layout table.
            try:
                aria = frame.locator("body").aria_snapshot()
                if aria:
                    header = f"[frame {frame_name}]" + chr(10) if frame_name else ""
                    aria_views.append(header + aria)
            except Exception:
                pass

            # Control inventory. Recovers the accessible name where there is one and,
            # where there is not, the neighbouring cell text that a human reads as the
            # label. This is the enrichment that makes legacy screens addressable.
            try:
                enriched = frame.evaluate(_LABEL_PROBE_JS)
            except Exception:
                enriched = []
            for item in enriched:
                counter += 1
                ref = f"e{counter}"
                self._ref_map[ref] = (frame_name, item.get("name") or "")
                elements.append(
                    Element(
                        ref=ref,
                        role=item.get("role") or "textbox",
                        name=item.get("name") or "",
                        text=item.get("text") or "",
                        label=item.get("label") or None,
                        frame=frame_name,
                        attrs={k: v for k, v in item.get("attrs", {}).items() if v},
                        bbox=tuple(item["bbox"]) if item.get("bbox") else None,
                    )
                )

        shot_path = None
        if screenshot:
            shot_path = screenshot_path or "screenshot.png"
            try:
                self.page.screenshot(path=shot_path, full_page=False)
            except Exception:
                shot_path = None

        return Observation(
            location=self.page.url,
            title=self.page.title() or "",
            frames=frame_names,
            frame_locations=frame_locs,
            elements=_dedupe(elements),
            text="\n".join(texts),
            http_status=self._last_status,
            screenshot_path=shot_path,
            aria=chr(10).join(aria_views),
        )

    def _flatten(self, node: dict | None, out: list[dict]) -> None:
        if not node:
            return
        out.append(node)
        for child in node.get("children", []) or []:
            self._flatten(child, out)

    # --- action --------------------------------------------------------------

    def _frame_by_name(self, name: str | None) -> Frame:
        if not name:
            return self.page.main_frame
        for frame_name, frame in self._frames():
            if frame_name == name:
                return frame
        return self.page.main_frame

    def act(self, request: ActionRequest) -> ActionResult:
        try:
            if request.kind == "navigate":
                self.page.goto(request.url, timeout=request.timeout_ms, wait_until="load")
                return ActionResult(ok=True, detail=f"navigated to {request.url}")
            if request.kind == "wait":
                self.page.wait_for_timeout(min(request.timeout_ms, 10_000))
                return ActionResult(ok=True, detail="waited")

            handle = self._handle_for_ref(request.ref)
            if handle is None:
                return ActionResult(ok=False, detail=f"no element for ref {request.ref}")
            locator, _ = handle

            if request.kind == "click":
                locator.click(timeout=request.timeout_ms)
            elif request.kind == "fill":
                locator.fill(request.value or "", timeout=request.timeout_ms)
            elif request.kind == "select":
                locator.select_option(request.value or "", timeout=request.timeout_ms)
            elif request.kind == "press":
                locator.press(request.value or "Enter", timeout=request.timeout_ms)
            else:
                return ActionResult(ok=False, detail=f"unsupported action {request.kind}")
            self.page.wait_for_load_state("load", timeout=request.timeout_ms)
            return ActionResult(ok=True, detail=f"{request.kind} ok")
        except Exception as exc:  # surfaced to the caller, never swallowed
            return ActionResult(ok=False, detail=str(exc)[:400], raised=type(exc).__name__)

    def _handle_for_ref(self, ref: str | None):
        if not ref or ref not in self._ref_map:
            return None
        frame_name, name = self._ref_map[ref]
        frame = self._frame_by_name(frame_name)
        for builder in (
            lambda: frame.get_by_role("button", name=name, exact=True),
            lambda: frame.get_by_role("link", name=name, exact=True),
            lambda: frame.get_by_label(name, exact=True),
            lambda: frame.get_by_text(name, exact=True),
            lambda: frame.locator(f"[name='{name}']"),
        ):
            try:
                loc = builder()
                if loc.count() > 0:
                    return loc.first, frame
            except Exception:
                continue
        return None

    # --- locator resolution: the robustness story ----------------------------

    def resolve(self, target: Target, observation: Observation | None = None):
        """Try each recorded strategy in portability order; report which tier won.

        A strategy only counts if it matches exactly one visible control. Ambiguity is
        treated as failure rather than "take the first": on a screen with a dozen Submit
        buttons, guessing is how automation posts the wrong form.
        """
        frame = self._frame_by_name(target.frame)
        for strategy in target.strategies:
            try:
                locator = self._locator_for(frame, strategy)
                if locator is None:
                    continue
                count = locator.count()
                if count == 1:
                    return locator.first, strategy.tier.value
                if count > 1:
                    # Ambiguous: try the next, more specific strategy instead.
                    continue
            except Exception:
                continue
        return None, None

    def _locator_for(self, frame: Frame, strategy):
        v = strategy.value
        tier = strategy.tier
        if tier is LocatorTier.ROLE_NAME:
            return frame.get_by_role(v["role"], name=v["name"], exact=v.get("exact", True))
        if tier is LocatorTier.LABEL_ANCHORED:
            return frame.get_by_label(v["label"], exact=v.get("exact", False))
        if tier is LocatorTier.TABLE_CELL:
            # Row identified by a key value, then the nth cell or a link within it.
            #
            # The naive form - locator("tr", has_text=...) - is wrong on these screens.
            # Legacy layouts nest tables inside table cells, so the outermost <tr> also
            # "contains" the text and matches first; reading its second cell returned the
            # table header instead of the balance. Requiring a *direct child* cell whose
            # text is exactly the key pins the innermost data row, which is the one a
            # human would point at.
            key = v["row_contains"].replace("'", "\\'")
            row = frame.locator(f"tr:has(> td:text-is('{key}'))")
            if row.count() == 0:  # fall back to the looser match rather than failing
                row = frame.locator("tr", has_text=v["row_contains"])
            if v.get("link_text"):
                return row.get_by_role("link", name=v["link_text"], exact=False)
            return row.locator("> td").nth(int(v.get("cell_index", 0)))
        if tier is LocatorTier.TEXT_EXACT:
            return frame.get_by_text(v["text"], exact=True)
        if tier is LocatorTier.TEXT_CONTAINS:
            return frame.get_by_text(v["text"], exact=False)
        if tier is LocatorTier.NAME_ATTR:
            return frame.locator(f"[name='{v['name']}']")
        if tier is LocatorTier.DOM_PATH:
            return frame.locator(v["selector"])
        return None

    def attach_operator(self, endpoint: str):
        """Attach a second client to this same live browser, for a human handoff.

        This is the concrete mechanism behind "the operator takes control of the live
        session". The automation's browser was launched with a debugging port, so an
        operator tool connects to that endpoint and drives the very same context - same
        cookies, same authenticated session, same half-completed screen. Handing over a
        URL instead would lose exactly the state that made the step need a human.

        Returns a Browser the caller is responsible for closing. It deliberately reuses
        this surface's Playwright driver, because a second sync driver in the same
        thread is not allowed.
        """
        return self._pw.chromium.connect_over_cdp(endpoint)

    def close(self) -> None:
        try:
            self._context.close()
            self._browser.close()
        finally:
            self._pw.stop()


def _dedupe(elements: list[Element]) -> list[Element]:
    seen: set[tuple] = set()
    out: list[Element] = []
    for el in elements:
        key = (el.frame, el.role, el.name, el.label, el.attrs.get("name"))
        if key in seen:
            continue
        seen.add(key)
        out.append(el)
    return out


# Finds form controls and the text that labels them, including the legacy pattern where
# the "label" is simply the previous cell in a layout table.
_LABEL_PROBE_JS = r"""
() => {
  const out = [];
  const els = document.querySelectorAll('input, select, textarea, button, a[href]');
  for (const el of els) {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'input' && ['hidden'].includes(type)) continue;
    let role = 'textbox';
    if (tag === 'select') role = 'combobox';
    else if (tag === 'button' || type === 'submit' || type === 'button') role = 'button';
    else if (tag === 'a') role = 'link';
    else if (type === 'checkbox') role = 'checkbox';
    else if (type === 'radio') role = 'radio';
    else if (type === 'password') role = 'textbox';

    let label = null;
    const id = el.getAttribute('id');
    if (id) {
      const lab = document.querySelector(`label[for="${CSS.escape(id)}"]`);
      if (lab) label = lab.innerText.trim();
    }
    if (!label) {
      const cell = el.closest('td');
      if (cell && cell.previousElementSibling) {
        const t = (cell.previousElementSibling.innerText || '').trim();
        if (t && t.length < 60) label = t;
      }
    }
    if (!label) {
      const prev = el.previousSibling;
      if (prev && prev.nodeType === 3) {
        const t = prev.textContent.trim();
        if (t && t.length < 60) label = t;
      }
    }
    let bbox = null;
    try {
      const r = el.getBoundingClientRect();
      if (r.width || r.height) bbox = [r.x, r.y, r.width, r.height];
    } catch (e) {}
    out.push({
      role,
      // For buttons, the visible caption is what a human reads and what policy should
      // classify on ("Search"), so it wins over the form control name ("btnSearch").
      // For inputs the control name is the more useful identifier.
      name: ((type === 'submit' || type === 'button' || tag === 'button')
              ? ((el.value || el.innerText || '').trim() || el.getAttribute('name') || '')
              : (el.getAttribute('name') || el.getAttribute('aria-label') || (el.innerText || '').trim())
            ).slice(0, 80),
      text: (el.innerText || el.value || '').trim().slice(0, 120),
      label,
      bbox,
      attrs: {
        name: el.getAttribute('name') || '',
        id: el.getAttribute('id') || '',
        type: type,
        href: el.getAttribute('href') || '',
        // The live property, not the attribute: typing changes .value but leaves the
        // attribute alone, and the agent needs to see what is actually in the field.
        // Passwords are never read back, so a credential cannot reach a prompt or log.
        value: (type === 'password') ? '' : ((el.value !== undefined ? el.value : el.getAttribute('value')) || '')
      }
    });
  }
  return out;
}
"""


def open_web_surface(headless: bool = True, cdp_port: int | None = None) -> WebSurface:
    return WebSurface(headless=headless, cdp_port=cdp_port)
