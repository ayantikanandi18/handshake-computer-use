# Computer-use capability system

An LLM works out how to do a task in a legacy back-office application once. What it
learned becomes a typed, versioned, reviewable capability. After that the task runs
deterministically, with no model in the loop, and returns a typed result that a calling
agent can branch on.

```
goal ──► discovery (LLM drives a live UI) ──► capability artifact ──► deterministic replay
                       │                                                      │
                       └──────────── escalate to a human ─────────────────────┘
                                   (same live session, lease transfer)
```

Everything below was built and run end to end. The discovery run is real: a local Llama
driving a real browser against a real application, with the evidence committed in
[`evidence/`](evidence/).

---

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/playwright install chromium
ollama serve & ollama pull qwen2.5:7b-instruct     # any OpenAI-compatible endpoint also works

python target_app/app.py &                          # the legacy app, on :8099

# 1. discovery: the model works the flow out for the first time
python -m cua.cli discover --goal "Sign on ... look up member 12345 ..." --member-id 12345

# 2. replay: same flow, no model, different parameter
python -m cua.cli replay --capability capabilities/meridian.member.read_savings_balance.v1.json \
                         --params '{"member_id":"22841"}'

# 3. every runtime condition the flow has to survive
python scripts/demo_matrix.py

# 4. human takes over the live session and hands it back
python scripts/demo_escalation.py

pytest -q      # 25 tests, no browser, no model
```

---

## What actually runs

### The discovery run

`qwen2.5:7b-instruct`, running locally, reached the goal in 8 steps:

```
1 fill  "txtUser"        2 fill  "txtPass"       3 click "Sign On"
4 click "Member Search"  5 fill  "txtMemberId"   6 click "Search"
7 click "View"           8 done
→ capabilities/meridian.member.read_savings_balance.v1.json
```

Evidence: [`evidence/discovery-004/`](evidence/discovery-004/) — per-step screenshots, the
decision log with the model's stated reasoning, and the policy verdict on every action.

### Deterministic replay against every runtime condition

`scripts/demo_matrix.py` runs the same artifact through the errors a production flow
actually meets. All eight behave as specified:

| Scenario | Result | Detail |
|---|---|---|
| happy path | `success` | `$ 4,182.55` |
| different member (parameterised) | `success` | `$ 15,904.12` |
| member does not exist | `business_outcome` | `MEMBER_NOT_FOUND` |
| restricted member | `business_outcome` | `ACCESS_DENIED` |
| surprise interstitial | `success` | recovered via `MAINTENANCE_INTERSTITIAL` |
| transient slow load | `success` | waited it out |
| session expires mid-flow | `success` | re-authenticated, then finished |
| application returns 500 | `failure` | `surface_error` |

The distinction that matters: rows 3 and 4 are **not failures**. "No such member" is the
answer to the question, and the calling agent gets it as data.

### The human handoff

`scripts/demo_escalation.py` — a policy-gated step stops the run, and an operator
finishes it **in the same browser session**:

```
control before run: automation
replay stopped    : escalated (step classified write; run limited to read)
control now       : nobody
  claimed by      : alice.operator  -> control = operator
  operator sees   : http://127.0.0.1:8099/app        # the authenticated session
  operator did    : clicked View on the result row
  released        -> control = automation
automation resumes on the operator's page: Share Balances visible = True
outputs after handoff: {'savings_balance': '$ 4,182.55'}
```

---

## The five decisions worth defending

### 1. The artifact is a contract, not a recording

[`src/cua/artifact/schema.py`](src/cua/artifact/schema.py)

A capability has to satisfy three readers at once: the calling agent wants a typed
function signature, the replay engine wants executable detail, and a reviewer wants to
decide whether this is safe to point at a core banking system. So the artifact declares
inputs, outputs, the closed set of business outcomes, per-step risk, and a checkpoint —
and deliberately excludes the model transcript, so replay can never come to depend on
the model's prose.

**Business outcomes are declared, not inferred.** If the agent above is going to branch
on `MEMBER_NOT_FOUND`, that set must be closed, typed, and reviewed before the
capability ships. Inferring it at runtime means every caller parses error strings.

### 2. Targets are layered strategies, ordered by portability

One selector is one point of failure, and on these screens it is usually the wrong one.
The generated ids look stable and are not: `ctl00_MainContent_gv1_ctl02_lnkView` encodes
the row's *position*, so it silently changes when a row is inserted above it. The
recorder explicitly rejects ids of that shape.

So each target carries every strategy the control supports, most portable first:

```
role_name → label_anchored → table_cell → text_exact → name_attr → dom_path
```

The ordering is a portability judgement. `role_name` and `label_anchored` describe a
control the way a person would — "the Search button", "the box next to Member ID" —
which is what survives a reskin, a version bump, and the jump to a desktop surface where
there is no DOM at all.

**Replay reports which tier matched,** and that is the cheapest drift detector available:
a flow that used to match on `role_name` and now only matches on `dom_path` is telling
you the app moved under it, well before it breaks.

Ambiguity is treated as failure, never "take the first". On a screen with a dozen Submit
buttons, guessing is how automation posts the wrong form.

### 3. Perception is the operator's view, not the DOM

[`src/cua/surface/base.py`](src/cua/surface/base.py)

The observation model is a list of controls with a role, a name, nearby text, and a
position. That is available from an accessibility tree, a DOM, a native automation API,
and — with OCR — from pixels. Building on it is what keeps the artifact honest about
what it depends on.

The target app demonstrates why this matters. Its ARIA tree for the login screen is:

```
row "Operator ID":
  cell "Operator ID"
  cell:
    - textbox          ← no accessible name at all
```

The input has no name. The meaning lives in the *adjacent cell*. The perception layer
recovers it, which is exactly what the `label_anchored` tier then depends on.

### 4. Policy is enforced at the action, and fails closed

[`src/cua/safety/policy.py`](src/cua/safety/policy.py)

The model is untrusted input — not malicious, but a text generator that will eventually
decide the right move is to click "Post Transactions". Every action is authorised before
it executes, in discovery *and* replay, against an explicit allowlist.

Risk is a property of the **step**, not the capability, because a read flow and a write
flow share most of their steps. Three classes: `read` runs, `write` needs the capability
to be approved, `irreversible` is never auto-approved — even for an approved capability.

**An unrecognised control is classified `write`, not `read`.** In a bank back office the
cost of wrongly assuming a button is harmless dwarfs the cost of stopping to ask.

Two things this caught while building, which are in the commit history rather than
retold here: the classifier blocked a read-only search because `\bsearch\b` does not
match `btnSearch` (fixed by normalising legacy camelCase control names), and replay
initially trusted the runtime heuristic over the artifact's reviewed risk (fixed by
taking the more conservative of the two).

### 5. Control transfer is a lease, not a flag

[`src/cua/escalation/broker.py`](src/cua/escalation/broker.py)

"The same live session, not a fresh one" rules out the easy version — screenshot the
failure, open a ticket, let a human redo it elsewhere. By the time a person starts over,
the state that made the problem interesting is gone.

So exactly one holder at a time, with explicit transfer:

```
AUTOMATION --raise--> NOBODY --claim--> OPERATOR --release--> AUTOMATION
```

A boolean `paused` flag would not answer "who is in control right now", and nothing
would stop two actors typing into the same form. The lease also expires, so an operator
who wanders off cannot strand the session.

The operator attaches over CDP to the browser the automation was already driving — same
cookies, same authenticated session, same half-completed screen. The console is mocked
(the brief allows it); the mechanism underneath is not.

---

## Designing for heterogeneity and scale

### Extending to legacy web, desktop, and terminal

The seam is [`Surface`](src/cua/surface/base.py): `observe()`, `act()`, `resolve()`.
Nothing above it imports Playwright. A new surface implements three methods.

| Surface | `observe()` | `act()` | Tiers that carry over |
|---|---|---|---|
| Modern web | DOM + ARIA | Playwright | all |
| **Legacy web (built)** | ARIA + label recovery, per frame | Playwright | all except stable ids |
| Desktop (Win32/WPF) | UIA tree → same `Element` shape | UIA invoke patterns | `role_name`, `label_anchored`, `bbox` |
| Terminal (3270/5250) | screen buffer → row/col cells | keystroke injection | `table_cell`, `text_exact` |

`Element` already carries `bbox`, and `LocatorStrategy.value` is an open dict, so a
desktop surface can record `AutomationId` or a terminal surface a screen cell without a
schema change. The artifact, replay engine, policy, and escalation are untouched.

The honest limit: `role_name` and `label_anchored` port well; `name_attr` and `dom_path`
are web-only and would simply never be recorded on another surface. A capability recorded
on web is **not** expected to replay on desktop — `SurfaceRequirement` makes replay refuse
rather than half-work. What ports is the *schema, engine, and policy*, which is where the
cost actually is.

### Hundreds of tenants running the same vendor product

Re-recording per tenant does not scale and throws away every shared improvement. So one
base capability plus a **tenant overlay** carrying only the delta:

```python
TenantOverlay(
    tenant_id="cu_north",
    entry_url="https://north.example/core/",
    target_overrides={"s04_click": Target(description="Find Member", ...)},
    extra_steps_after={"s04_click": [Step(...)]},   # a consent screen only some tenants have
)
```

`capability.resolved_for(tenant)` is pure — the base is never mutated, so a reviewer can
diff base against resolved and see exactly what one tenant changed.

**Drift detection falls out of the tier reporting.** Every replay records which tier
matched. Aggregated per tenant and version, that is a cheap early-warning signal:

- all tenants slide to a lower tier on the same step → the vendor changed the product;
  fix the base.
- one tenant slides → that tenant's configuration drifted; add an overlay.
- checkpoint failures cluster on one version → gate that version in `version_range`.

This is designed and the overlay mechanism is implemented and tested; the aggregation
service is not built.

---

## What I cut, and what I would do next

Deliberately thin, at a real seam:

- **Operator console** is a script, not a UI. The lease, CDP attach, audit trail, and
  resume are real; the front end is not the interesting part.
- **Desktop and terminal surfaces** are designed, not built. The `Surface` protocol is
  the proof the seam exists.
- **Multi-tenant** overlay resolution is implemented and tested; drift aggregation is
  design only.
- **Business outcomes and recoverable conditions are hand-written**, in
  [`src/cua/cli.py`](src/cua/cli.py). A single happy-path discovery run never sees an
  error screen, so a recorder claiming to have discovered them would be lying. This is
  the reviewed part of the capability and the artifact should carry a review flag.

With more time, in priority order:

1. **Error-space discovery.** Drive discovery deliberately into failure states and
   propose business outcomes for a human to confirm. This is the biggest gap.
2. **Self-healing with a bounded model call.** When replay hits `target_not_found`, it
   currently stops or escalates. It could re-observe, ask the model *only* to re-identify
   that one control, and propose a locator patch for review — keeping the model out of
   the decision loop while using it for the one thing it is good at.
3. **Capability versioning and promotion.** Record → review → approve → pin, with replay
   refusing unapproved capabilities in production.
4. **Concurrency.** One session per process today. Real deployment needs a session pool,
   and the lease model is already the right primitive.

### Known rough edges

- The 7B local model needed a strict action grammar, a scratchpad, and per-frame
  locations before it could finish the flow. A frontier model would need less scaffolding
  — but building against the weak one forced a stricter contract, which is a better
  design. Swapping providers is one environment variable.
- `_extract_outputs` is called directly in the escalation demo to show post-handoff
  extraction; a resumable replay (continue from step N) is the right shape and is not
  built.
- Output extraction supports text, with `strip` and `currency_to_number` transforms.
  Attribute extraction is declared in the schema but unused.
