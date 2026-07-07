"""FastAPI gateway exposing the FastMCP tools as HTTP endpoints (PLAN §3).

Run with:  uvicorn src.tools.server:app --reload
or:        python run_api.py

The endpoints are thin wrappers over the pure tool functions; the same
functions are called directly by the agents, so the server is optional.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .. import data_loader
from . import (
    evaluate_drug,
    evaluate_surgery,
    generate_symptoms,
    lookup,
    simulate_labs,
)

app = FastAPI(
    title="MedSim FastMCP Tool Gateway",
    version="0.1.0",
    description="Clinical simulation tools: symptoms, labs, drugs, knowledge, surgery.",
)


def _require_disease(disease_id: str) -> dict:
    disease = data_loader.get_disease(disease_id)
    if disease is None:
        raise HTTPException(status_code=404, detail=f"Unknown disease_id '{disease_id}'")
    return disease


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class SymptomReq(BaseModel):
    disease_id: str


class LabReq(BaseModel):
    ordered_tests: List[str]
    disease_id: str


class DrugReq(BaseModel):
    drug: str
    disease_id: str
    patient_allergies: List[str] = []
    current_drugs: List[str] = []


class LookupReq(BaseModel):
    query: str


class SurgeryReq(BaseModel):
    procedure: str
    disease_id: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    return {"status": "ok", "diseases": len(data_loader.diseases())}


@app.get("/")
def root():
    return {
        "service": "MedSim FastMCP Tool Gateway",
        "endpoints": [
            "/api/v1/symptoms", "/api/v1/labs", "/api/v1/drug",
            "/api/v1/lookup", "/api/v1/surgery",
        ],
    }


@app.post("/api/v1/symptoms")
def api_symptoms(req: SymptomReq):
    return generate_symptoms(_require_disease(req.disease_id))


@app.post("/api/v1/labs")
def api_labs(req: LabReq):
    disease = _require_disease(req.disease_id)
    return simulate_labs(req.ordered_tests, disease)


@app.post("/api/v1/drug")
def api_drug(req: DrugReq):
    disease = _require_disease(req.disease_id)
    return evaluate_drug(req.drug, disease, req.patient_allergies, req.current_drugs)


@app.post("/api/v1/lookup")
def api_lookup(req: LookupReq):
    return lookup(req.query)


@app.post("/api/v1/surgery")
def api_surgery(req: SurgeryReq):
    disease = _require_disease(req.disease_id)
    return evaluate_surgery(req.procedure, disease)
