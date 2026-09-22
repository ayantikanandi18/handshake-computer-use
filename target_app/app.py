"""
Meridian Core - Member Servicing.

A stand-in for the kind of back-office application this system exists to automate.
It is deliberately hostile in the ways real bank software is hostile:

  * a frameset, so "the page" is not one document
  * table-based layout, non-semantic markup, no test IDs
  * server-generated control names like ctl00_MainContent_gv1_ctl02_lnkView that
    look stable but are positional and shift when rows move
  * labels that are not associated with their inputs
  * navigation via javascript: links and form posts rather than hrefs

It also exposes a fault-injection switchboard (/admin/faults). The brief is explicit
that the interesting replay failures are runtime conditions, not layout drift, so the
target has to be able to produce those conditions on demand and reproducibly. Without
that there is no honest way to demonstrate an error taxonomy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from flask import Flask, redirect, render_template, request, session, url_for

app = Flask(__name__)
app.secret_key = "meridian-core-demo-not-a-real-secret"  # demo only; never a real key


# --- fake book of record -----------------------------------------------------

@dataclass
class Member:
    member_id: str
    name: str
    status: str
    savings_balance: str
    checking_balance: str
    restricted: bool = False
    sub_accounts: list[str] = field(default_factory=list)


MEMBERS: dict[str, Member] = {
    "12345": Member("12345", "Dolores Abernathy", "Active", "4,182.55", "912.30"),
    "22841": Member("22841", "Bernard Lowe", "Active", "15,904.12", "2,301.88"),
    "31007": Member("31007", "Maeve Millay", "Dormant", "212.00", "0.00"),
    "44190": Member("44190", "Teddy Flood", "Active", "8,640.75", "1,150.00", restricted=True),
}


# --- fault injection ---------------------------------------------------------
# Each flag maps to one of the runtime conditions a production replay must survive.

FAULTS: dict[str, bool] = {
    "session_timeout": False,   # session expires mid-flow
    "permission_denied": False, # operator lacks entitlement for this record
    "surprise_dialog": False,   # unexpected interstitial before the content renders
    "slow_load": False,         # transient slowness, recoverable by waiting
    "app_error": False,         # outright 500
}


def _fault(name: str) -> bool:
    return FAULTS.get(name, False)


@app.route("/admin/faults", methods=["GET", "POST"])
def admin_faults():
    """Switchboard so tests and demos can make a specific condition happen on demand."""
    if request.method == "POST":
        for key in FAULTS:
            FAULTS[key] = request.form.get(key) == "on"
        if request.form.get("reset"):
            for key in FAULTS:
                FAULTS[key] = False
    return render_template("admin_faults.html", faults=FAULTS)


@app.route("/admin/faults/api", methods=["POST"])
def admin_faults_api():
    """Programmatic switchboard, so the test suite does not have to drive the UI."""
    payload = request.get_json(silent=True) or {}
    for key in FAULTS:
        if key in payload:
            FAULTS[key] = bool(payload[key])
    return {"faults": FAULTS}


# --- session plumbing --------------------------------------------------------

def _logged_in() -> bool:
    if _fault("session_timeout") and session.get("user"):
        # Expire exactly once so a replay can observe the timeout and recover by
        # re-authenticating, rather than being stuck in a redirect loop forever.
        session.pop("user", None)
        FAULTS["session_timeout"] = False
        return False
    return bool(session.get("user"))


def _guard():
    """Returns a response to short-circuit with, or None to continue."""
    if _fault("app_error"):
        return render_template("error_500.html"), 500
    if not _logged_in():
        return redirect(url_for("login", expired="1"))
    if _fault("slow_load"):
        time.sleep(4.0)  # transient: slow but it does eventually come back
    return None


# --- screens -----------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user = (request.form.get("txtUser") or "").strip()
        pwd = (request.form.get("txtPass") or "").strip()
        if user and pwd:
            session["user"] = user
            return redirect(url_for("shell"))
        error = "User ID and password are required."
    return render_template("login.html", error=error, expired=request.args.get("expired"))


@app.route("/app")
def shell():
    if not _logged_in():
        return redirect(url_for("login", expired="1"))
    return render_template("shell.html")  # the frameset


@app.route("/nav")
def nav():
    return render_template("nav.html")


@app.route("/content")
def content_home():
    guard = _guard()
    if guard:
        return guard
    return render_template("home.html", user=session.get("user"))


@app.route("/content/search", methods=["GET", "POST"])
def member_search():
    guard = _guard()
    if guard:
        return guard

    if _fault("surprise_dialog"):
        FAULTS["surprise_dialog"] = False  # fire once
        return render_template("interstitial.html", next_url=url_for("member_search"))

    results = None
    message = None
    if request.method == "POST":
        query = (request.form.get("txtMemberId") or "").strip()
        if not query:
            message = "Member ID is required."
        elif not query.isdigit():
            message = "Member ID must be numeric."
        else:
            member = MEMBERS.get(query)
            results = [member] if member else []
            if not results:
                message = "No member found matching that ID."
    return render_template("search.html", results=results, message=message)


@app.route("/content/member/<member_id>")
def member_detail(member_id: str):
    guard = _guard()
    if guard:
        return guard

    member = MEMBERS.get(member_id)
    if not member:
        return render_template("not_found.html", member_id=member_id), 404
    if member.restricted or _fault("permission_denied"):
        return render_template("denied.html", member_id=member_id), 403
    return render_template("member.html", m=member)


@app.route("/content/member/<member_id>/subaccount", methods=["GET", "POST"])
def sub_account(member_id: str):
    guard = _guard()
    if guard:
        return guard

    member = MEMBERS.get(member_id)
    if not member:
        return render_template("not_found.html", member_id=member_id), 404

    error = None
    if request.method == "POST":
        nickname = (request.form.get("txtNickname") or "").strip()
        deposit = (request.form.get("txtDeposit") or "").strip()
        acct_type = request.form.get("ddlType") or ""
        if not nickname:
            error = "Nickname is required."
        elif not acct_type:
            error = "Account type must be selected."
        else:
            try:
                amount = float(deposit or "0")
            except ValueError:
                amount = -1
            if amount < 25:
                error = "Opening deposit must be at least $25.00."
        if not error:
            ref = f"SA-{member_id}-{len(member.sub_accounts) + 1:03d}"
            member.sub_accounts.append(ref)
            return render_template(
                "subaccount_confirm.html", m=member, ref=ref,
                nickname=nickname, acct_type=acct_type, deposit=f"{float(deposit):,.2f}",
            )
    return render_template("subaccount_new.html", m=member, error=error)


if __name__ == "__main__":
    app.run(port=8099, debug=False)
