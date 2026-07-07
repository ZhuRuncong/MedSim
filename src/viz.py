"""Agent-routing graph as inline SVG.

Renders the multi-agent topology (Supervisor → specialised agents → tools) and
lights up the path the current case actually took, from the trajectory spans in
``GameState.trace``. Pure function → unit-testable and reused by the Streamlit UI.
"""
from __future__ import annotations

import html
from typing import List

from .trace import ROUTING, TraceEvent

# Palette (matches the academic UI; SVG can't read the page's CSS vars).
INK, MUTED, LINE = "#1b2733", "#5f6f7e", "#cbd4de"
ACCENT, ACCENT_SOFT, ACCENT_LINE = "#1f4e79", "#eef3f8", "#c2d3e5"
PANEL_ALT, LLM = "#f7f9fb", "#6b3fa0"

_AGENTS = [  # (trace agent name, label, y)
    ("PatientAgent", "Patient", 8),
    ("ClinicalAgent", "Clinical", 74),
    ("SurgeryAgent", "Surgery", 140),
    ("KnowledgeAgent", "Knowledge", 206),
    ("CriticAgent", "Critic", 272),
]
_TOOLS = [  # (tool name, owning agent, y)
    ("symptom_generator", "PatientAgent", 8),
    ("lab_simulator", "ClinicalAgent", 60),
    ("drug_effect_engine", "ClinicalAgent", 98),
    ("emergency_surgery", "SurgeryAgent", 140),
    ("knowledge_lookup", "KnowledgeAgent", 206),
]

_AX, _AW, _AH = 232, 150, 40          # agent column x, width, height
_TX, _TW, _TH = 452, 170, 30          # tool column
_SUP = (20, 150, 150, 46)            # supervisor x,y,w,h
_W, _H = 640, 322


def _rect(x, y, w, h, label, used, *, llm=False, count=0, tool=False):
    fill = ACCENT_SOFT if used else PANEL_ALT
    stroke = ACCENT if used else LINE
    tcol = ACCENT if used else MUTED
    fs = 12 if not tool else 11
    font = "ui-monospace,Consolas,monospace" if tool else "system-ui,sans-serif"
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
         f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if used else 1}"/>'
         f'<text x="{x + 12}" y="{y + h / 2 + 4}" font-family="{font}" '
         f'font-size="{fs}" fill="{tcol}" font-weight="600">{html.escape(label)}</text>')
    if count:
        cx, cy = x + w - 15, y + 14
        s += (f'<circle cx="{cx}" cy="{cy}" r="10" fill="{ACCENT}"/>'
              f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-family="system-ui" '
              f'font-size="10" fill="#fff" font-weight="700">{count}</text>')
    if llm:
        s += (f'<rect x="{x + 12}" y="{y + h - 15}" width="30" height="12" rx="3" fill="#f1ecfa" '
              f'stroke="#d9caf0"/><text x="{x + 27}" y="{y + h - 6}" text-anchor="middle" '
              f'font-family="system-ui" font-size="8" fill="{LLM}" font-weight="700">LLM</text>')
    return s


def _edge(x1, y1, x2, y2, used):
    col = ACCENT if used else LINE
    dash = "" if used else ' stroke-dasharray="3,3"'
    return (f'<path d="M {x1} {y1} C {x1 + 40} {y1}, {x2 - 40} {y2}, {x2} {y2}" '
            f'fill="none" stroke="{col}" stroke-width="{2 if used else 1}"{dash}/>')


def routing_graph_svg(trace: List[TraceEvent]) -> str:
    trace = trace or []
    agent_count, tool_count, llm_agents = {}, {}, set()
    for e in trace:
        if e.agent:
            agent_count[e.agent] = agent_count.get(e.agent, 0) + 1
            if e.llm_used:
                llm_agents.add(e.agent)
        for t in e.tools:
            tool_count[t] = tool_count.get(t, 0) + 1
    # symptom_generator runs at intake (not per-action), so it's used once a case exists.
    if trace:
        tool_count.setdefault("symptom_generator", 1)

    sx, sy, sw, sh = _SUP
    parts = [f'<svg viewBox="0 0 {_W} {_H}" xmlns="http://www.w3.org/2000/svg" '
             f'style="width:100%;height:auto;font-family:system-ui">']

    # edges first (under nodes)
    for name, _, ay in _AGENTS:
        parts.append(_edge(sx + sw, sy + sh / 2, _AX, ay + _AH / 2, name in agent_count))
    for tool, owner, ty in _TOOLS:
        parts.append(_edge(_AX + _AW, _agent_y(owner) + _AH / 2, _TX, ty + _TH / 2,
                           tool in tool_count))

    # nodes
    parts.append(_rect(sx, sy, sw, sh, "Supervisor", True, count=len(trace)))
    parts.append(f'<text x="{sx}" y="{sy - 8}" font-family="system-ui" font-size="10" '
                 f'fill="{MUTED}">router</text>')
    for name, label, ay in _AGENTS:
        parts.append(_rect(_AX, ay, _AW, _AH, label, name in agent_count,
                           llm=name in llm_agents, count=agent_count.get(name, 0)))
    for tool, _, ty in _TOOLS:
        parts.append(_rect(_TX, ty, _TW, _TH, tool, tool in tool_count, tool=True))

    parts.append(f'<text x="{_AX}" y="{_H - 4}" font-family="system-ui" font-size="10" '
                 f'fill="{MUTED}">agents</text>')
    parts.append(f'<text x="{_TX}" y="{_H - 4}" font-family="system-ui" font-size="10" '
                 f'fill="{MUTED}">tools</text>')
    parts.append("</svg>")
    return "".join(parts)


def _agent_y(agent_name: str) -> float:
    for name, _, ay in _AGENTS:
        if name == agent_name:
            return ay
    return 0


# Sanity: every routable action maps to an agent we draw.
_DRAWN = {a for a, _, _ in _AGENTS}
assert all(r["agent"] in _DRAWN for r in ROUTING.values()), "viz agents out of sync with ROUTING"
