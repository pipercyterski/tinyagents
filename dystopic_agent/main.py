"""Dystopic code-mode agent — a telemetry-instrumented port of TinyToolCallingAgent.

Ported from ``tinytoolcallingagent.py`` (upstream: albertvillanova/tinyagents).
The upstream agent is a ReAct tool-calling loop: take a query, let an LLM pick a
tool, execute it over MCP, observe the result, answer. This port keeps that
*shape* but:

  * swaps MCP for the Dystopic per-run proxy (``POST {proxy_url}/tools/{name}``),
    so tool responses come from the platform's simulated world, and
  * replaces the LLM decision with deterministic triage, so the run needs no
    agent-side model key (the platform's org model drives the simulated user and
    the judge) — the run stays cheap and deterministic.

The point of this agent is to exercise the *telemetry span lane*: it emits all
four span kinds (``note``, ``guardrail_decision``, ``state_snapshot``,
``handoff_traversal``), links some spans to the tool call they explain via
``step_ref``, and attributes the handoff sub-turn with ``actor_id``.

The platform imports this file and calls ``run(...)`` inside its E2B sandbox.
Everything here is stdlib-only on purpose — the sandbox does not ship the
dystopic SDK, and the ``/telemetry`` wire body is byte-identical to what the
SDK's ``emit()`` sends, so this is a faithful end-to-end exercise of the route,
storage, projection, and web render.
"""

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _post(url: str, body: dict, run_token: str, *, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {run_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def proxy_call(name: str, arguments: dict, *, proxy_url: str, run_token: str) -> dict:
    """Call a platform (simulated) tool through the per-run proxy.

    Returns the full proxy envelope ``{"tool_name", "response", "sequence"?}`` so
    the caller can read the tool call's ``sequence`` and thread it into a
    telemetry span's ``step_ref``.
    """
    return _post(
        f"{proxy_url.rstrip('/')}/tools/{name}",
        arguments,
        run_token,
        timeout=120.0,
    )


def emit(
    kind: str,
    payload: dict | None = None,
    *,
    proxy_url: str,
    run_token: str,
    step_ref: int | None = None,
    actor_id: str | None = None,
) -> int | None:
    """Emit one telemetry span. Never raises — a telemetry hiccup must not fail
    the run (mirrors the SDK's ``safe_emit`` posture)."""
    body: dict = {
        "kind": kind,
        "payload": payload or {},
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }
    if step_ref is not None:
        body["step_ref"] = step_ref
    if actor_id is not None:
        body["actor_id"] = actor_id
    try:
        reply = _post(
            f"{proxy_url.rstrip('/')}/telemetry", body, run_token, timeout=30.0
        )
        return reply.get("sequence")
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


def _tool_sequence(envelope: dict) -> int | None:
    """Best-effort extraction of a tool call's per-run sequence from the proxy
    envelope, for use as a telemetry ``step_ref``."""
    seq = envelope.get("sequence")
    return seq if isinstance(seq, int) else None


def run(task_input: dict, *, proxy_url: str, run_token: str) -> dict:
    """Platform entrypoint. Returns ``{"final_response", "metadata"}``."""
    instruction = (task_input.get("user_instruction") or "").strip()
    P = {"proxy_url": proxy_url, "run_token": run_token}

    # 1. Reasoning note — the plan (a `note` span, no tool anchor).
    emit(
        "note",
        {"text": "Triage v2: classify the request, run the guardrail, then search the KB.", "phase": "plan"},
        **P,
    )

    # 2. Input guardrail — a `guardrail_decision` span.
    lowered = instruction.lower()
    blocked = any(w in lowered for w in ("password", "ssn", "social security", "credit card"))
    emit(
        "guardrail_decision",
        {
            "decision": "block" if blocked else "allow",
            "rule_name": "input-pii-guard",
            "reason": "sensitive pattern detected" if blocked else "no sensitive patterns",
        },
        **P,
    )
    if blocked:
        emit("note", {"text": "Refused: input tripped the PII guard.", "phase": "final"}, **P)
        return {
            "final_response": (
                "I'm not able to help with sharing sensitive personal data like "
                "passwords or card numbers. Please contact support through a secure channel."
            ),
            "metadata": {"blocked": True},
        }

    # 3. Tool call — KB search (a real proxy/simulated tool → tool span).
    kb_env = proxy_call("search_docs", {"query": instruction[:200] or "help"}, **P)
    kb_answer = kb_env.get("response", kb_env)
    kb_seq = _tool_sequence(kb_env)

    # 4. State snapshot — anchored to the search tool call via `step_ref`.
    emit(
        "state_snapshot",
        {"snapshot": {"retrieved": kb_answer}, "label": "after-search_docs"},
        step_ref=kb_seq,
        **P,
    )

    # 5. Escalation path — a `handoff_traversal` span + a sub-agent-attributed
    #    tool call + note (exercises `actor_id` sub-agent attribution rendering).
    wants_human = any(w in lowered for w in ("refund", "human", "agent", "escalate", "manager"))
    if wants_human:
        emit(
            "handoff_traversal",
            {"from": "triage", "to": "billing-specialist", "reason": "human/refund escalation"},
            actor_id="billing-specialist",
            **P,
        )
        ticket_env = proxy_call(
            "create_ticket",
            {"summary": instruction[:120] or "customer request", "priority": "high"},
            **P,
        )
        ticket = ticket_env.get("response", ticket_env)
        emit(
            "note",
            {"text": f"Opened escalation ticket: {ticket}", "phase": "handoff"},
            step_ref=_tool_sequence(ticket_env),
            actor_id="billing-specialist",
            **P,
        )
        final = (
            "I've escalated this to our billing specialists and opened a priority "
            f"ticket for you: {ticket}. They'll follow up shortly."
        )
    else:
        final = f"Here's what I found in our docs: {kb_answer}"

    # 6. Final reasoning note.
    emit("note", {"text": "Composed the final response.", "phase": "final"}, **P)

    return {
        "final_response": final,
        "metadata": {"kb": kb_answer, "escalated": wants_human},
    }
