"""lab_simulator — POST /api/v1/labs.

Pull normal ranges for the ordered panels, generate plausible values, then
overlay disease-specific deviations (e.g. ↑WBC in pneumonia). Qualitative /
imaging studies return a textual finding. PLAN §3.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .. import data_loader

# Default (normal) reads for qualitative studies when the disease has no
# specific finding for that test.
_NORMAL_QUALITATIVE = {
    "Chest X-ray": "No focal consolidation, effusion or pneumothorax.",
    "ECG": "Normal sinus rhythm; no acute ST-T changes.",
    "Abdominal Ultrasound": "No gallstones; normal biliary tree and solid organs.",
    "CT Abdomen/Pelvis": "No acute intra-abdominal abnormality.",
    "Transvaginal Ultrasound": "Normal-appearing uterus and adnexa; no free fluid.",
    "Pelvic Ultrasound": "Normal pelvic organs; no free fluid.",
    "RSV Antigen": "Negative for respiratory syncytial virus.",
    "Rapid Strep Test": "Negative for group A streptococcus.",
    "Blood Culture": "No growth to date.",
    "Sputum Culture": "Mixed oral flora; no predominant pathogen.",
    "Urine Culture": "No significant growth.",
    "Wound Culture": "No significant growth.",
    "Skin Swab": "No significant bacterial growth.",
    "Urinalysis": "Clear; negative leukocyte esterase, nitrites, blood, protein and ketones.",
    "Serum Ketones": "Beta-hydroxybutyrate within normal limits.",
    "Blood Type & Screen": "Blood type O positive; antibody screen negative.",
    "Urine Protein/Creatinine Ratio": "Within normal limits (<0.3 mg/mg).",
}


def _round(value: float, low: float, high: float):
    if abs(value) >= 100:
        return int(round(value))
    span = high - low
    if span <= 0.5:
        return round(value, 2)
    if high <= 20:
        return round(value, 1)
    return int(round(value))


def _flag(value: float, low: float, high: float) -> str:
    if value < low:
        return "L"
    if value > high:
        return "H"
    return "N"


def _normal_value(comp: dict, rng: np.random.Generator) -> float:
    low, high = comp["low"], comp["high"]
    lo = low + 0.20 * (high - low)
    hi = high - 0.20 * (high - low)
    return float(rng.uniform(lo, hi))


def _qualitative(test: str, disease: dict, vitals: dict, deviations: dict) -> dict:
    if test == "Pulse Oximetry":
        spo2 = vitals.get("SpO2")
        finding = (f"SpO2 {spo2}% on room air." if spo2 is not None
                   else "SpO2 within normal limits.")
        return {"type": "qualitative", "finding": finding,
                "flag": "L" if (spo2 is not None and spo2 < 94) else "N"}
    dev = deviations.get(test)
    if dev and "finding" in dev:
        return {"type": "qualitative", "finding": dev["finding"], "flag": "abnormal"}
    return {"type": "qualitative",
            "finding": _NORMAL_QUALITATIVE.get(test, "No acute abnormality detected."),
            "flag": "N"}


def simulate_labs(
    ordered_tests: List[str],
    disease: dict,
    vitals: Optional[dict] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    vitals = vitals or disease.get("vitals", {})
    deviations = disease.get("lab_deviations", {})
    citation = disease.get("guideline", {}).get("url")

    results: Dict[str, dict] = {}
    for test in ordered_tests:
        if test in data_loader.QUALITATIVE_TESTS:
            results[test] = _qualitative(test, disease, vitals, deviations)
            continue

        comps = data_loader.components_for_test(test)
        if not comps:
            results[test] = {"type": "unavailable",
                             "note": f"'{test}' is not available at this facility."}
            continue

        panel = {}
        for comp in comps:
            name = comp["component"]
            dev = deviations.get(name)
            span = comp["high"] - comp["low"]
            forced = None
            if dev and "value" in dev:
                value = float(dev["value"])
            elif dev and dev.get("direction") == "high":
                value = comp["high"] + 0.6 * (span or abs(comp["high"]) or 1.0)
                forced = "H"
            elif dev and dev.get("direction") == "low":
                value = max(comp["low"] - 0.6 * (span or 1.0), 0.0)
                forced = "L"
            else:
                value = _normal_value(comp, rng)
            value = _round(value, comp["low"], comp["high"])
            # For directional deviations, keep the intended abnormal flag even if
            # rounding snaps the display value back onto the reference boundary.
            panel[name] = {
                "value": value,
                "unit": comp["unit"],
                "ref_low": comp["low"],
                "ref_high": comp["high"],
                "flag": forced or _flag(value, comp["low"], comp["high"]),
            }
        results[test] = {"type": "panel", "components": panel}

    return {
        "disease_id": disease.get("id"),
        "results": results,
        "citations": [c for c in [citation] if c],
    }
