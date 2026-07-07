"""Shared game state (LangGraph §4).

`GameState` is the single object every agent reads & writes. It is a Pydantic
model so it serialises cleanly to/from Redis (or the in-memory fallback store).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .trace import TraceEvent


class ScoreEvent(BaseModel):
    """One traceable point change (mirrors what LangSmith would capture)."""

    action: str                       # rubric key, e.g. "order_appropriate_test"
    delta: int
    agent: str                        # which agent awarded it
    reason: str = ""
    citation: Optional[str] = None    # guideline URL / DOI for transparency
    turn: int = 0


class Message(BaseModel):
    """One entry in the case feed shown to the student."""

    role: str                         # patient | supervisor | clinical | ...
    text: str
    kind: str = "info"                # info | gain | loss | success | fail
    points_delta: int = 0
    citations: List[str] = Field(default_factory=list)
    turn: int = 0
    llm: bool = False                 # was this text voiced/synthesised by an LLM?


class Debrief(BaseModel):
    """Structured end-of-case after-action report (the DebriefAgent's output)."""

    outcome: str                       # "correct" | "missed"
    true_diagnosis: str
    attempts: int = 0
    turns: int = 0
    points: int = 0
    max_points: int = 0
    efficiency: float = 0.0            # points / ideal, clamped 0..1

    tests_hit: List[str] = Field(default_factory=list)
    tests_missed: List[str] = Field(default_factory=list)
    tests_low_value: List[str] = Field(default_factory=list)
    exams_hit: List[str] = Field(default_factory=list)
    exams_missed: List[str] = Field(default_factory=list)
    drugs_first_line: List[str] = Field(default_factory=list)
    drugs_missed_first_line: List[str] = Field(default_factory=list)
    drugs_harmful: List[str] = Field(default_factory=list)

    surgery_note: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    teaching_points: List[str] = Field(default_factory=list)
    safety_flags: List[str] = Field(default_factory=list)
    attending_note: str = ""
    citation: Optional[str] = None
    llm: bool = False


class GameState(BaseModel):
    # --- Core patient data (fixed per case) -------------------------------- #
    case_id: str
    disease_id: str
    disease_name: str                 # hidden ground truth (not shown to student)
    specialty: str                    # the specialty this case was drawn from
    chief_complaint: str = ""
    demographics: Dict[str, str] = Field(default_factory=dict)  # age, sex, ...
    allergies: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)           # generated at start
    vitals: Dict[str, float] = Field(default_factory=dict)      # HR, BP, RR, Temp, SpO2

    # --- Student identity / progression ------------------------------------ #
    student_id: str = "guest"
    student_name: str = "Student"
    level: int = 1

    # --- Dynamic interaction history --------------------------------------- #
    asked_history: List[str] = Field(default_factory=list)
    performed_exam: List[str] = Field(default_factory=list)
    ordered_tests: List[str] = Field(default_factory=list)
    test_results: Dict[str, Dict] = Field(default_factory=dict)
    prescribed_drugs: List[str] = Field(default_factory=list)
    surgery_requested: Optional[str] = None
    surgeries: List[str] = Field(default_factory=list)
    diagnosis: Optional[str] = None
    diagnosis_differentials: List[str] = Field(default_factory=list)  # ranked, metacognition
    diagnosis_confidence: Optional[int] = None                        # 0-100

    # --- Scoring & flow control -------------------------------------------- #
    points: int = 0
    turn: int = 0
    retries: int = 0                  # incorrect-diagnosis attempts used
    max_retries: int = 3
    status: str = "active"            # active | complete | failed
    gates_fired: List[str] = Field(default_factory=list)  # safety gates already penalised

    # --- Traceability / feed ----------------------------------------------- #
    score_log: List[ScoreEvent] = Field(default_factory=list)
    feed: List[Message] = Field(default_factory=list)
    trace: List[TraceEvent] = Field(default_factory=list)  # agent trajectory
    debrief: Optional[Debrief] = None                      # end-of-case report

    # ------------------------------------------------------------------ #
    def redis_key(self) -> str:
        return f"session:{self.student_id}:{self.case_id}"
