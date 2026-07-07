"""Ground the drug reference data against real-world sources.

Rewrites data/drug_interactions.json in place, grounding each drug's class /
ATC / monograph / contraindications / label URL (RxNorm + openFDA, both free)
and rebuilding the interaction matrix from DDInter 2.0 + an ONC high-priority
override, merged with the hand-curated pairs (curated descriptions kept; ONC
forces 'major'). The app stays local-first: this runs offline once and commits
static, grounded data — no live API calls during play.

Usage:
  python scripts/ground_drug_data.py            # ground everything
  python scripts/ground_drug_data.py --offline  # ONC overrides only (no network)
  python scripts/ground_drug_data.py --no-formulary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:  # Windows consoles default to cp1252; avoid UnicodeEncodeError crashes.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import config, grounding  # noqa: E402


def log(m):
    print(f"[ground] {m}")


def ground_formulary(db: dict) -> dict:
    drugs = db.get("drugs", {})
    n_class = n_label = n_atc = 0
    warnings = []
    for name, meta in drugs.items():
        if meta.get("class") == "diagnostic":
            continue
        rx = grounding.rxclass_for_name(name)
        time.sleep(0.1)  # be polite to RxNav (20 req/s cap)
        fda = grounding.openfda_label_for_name(name)
        sources = list(meta.get("sources", []))

        if rx.get("class"):
            meta["class"] = rx["class"]; n_class += 1; sources.append("rxnorm")

        # Choose the ATC that best fits: prefer one matching the curated allergy
        # family (disambiguates combo-product ATCs), else the most-specific.
        cur = meta.get("allergy_class")
        atcs = rx.get("atcs") or []
        chosen = None
        if cur not in (None, "none"):
            chosen = next((a for a in atcs if grounding.atc_to_allergy_family(a) == cur), None)
        chosen = chosen or rx.get("atc")
        if chosen:
            meta["atc"] = chosen; n_atc += 1
            fam = grounding.atc_to_allergy_family(chosen)
            if fam:
                meta["atc_allergy_family"] = fam
                if cur in (None, "none"):
                    meta["allergy_class"] = fam            # fill only if missing
                elif cur != fam:                            # never silently override a curated safety value
                    warnings.append(f"{name}: curated allergy_class={cur!r} vs ATC={fam!r} (kept curated)")
        for k in ("monograph", "contraindications", "boxed_warning", "label_url", "rxcui"):
            if fda.get(k):
                meta[k] = fda[k]
        if any(fda.get(k) for k in ("monograph", "label_url")):
            n_label += 1; sources.append("openfda")

        if sources:
            meta["grounded"] = True
            meta["sources"] = sorted(set(sources))
    log(f"formulary grounded: {n_class} classes, {n_atc} ATC codes, {n_label} openFDA labels "
        f"({len(drugs)} drugs)")
    for w in warnings:
        log("  ! " + w)
    return db


def ground_interactions(db: dict, offline: bool) -> dict:
    curated = db.get("interactions", {})
    n_curated = sum(len(v) for v in curated.values())
    ddinter = {}
    if not offline:
        log("downloading DDInter shards...")
        rows = grounding.download_ddinter()
        if rows:
            name_map = grounding.ddinter_name_map(list(db.get("drugs", {})))
            ddinter = grounding.parse_ddinter(rows, name_map)
            log(f"DDInter: {len(rows)} rows -> {sum(len(v) for v in ddinter.values())} pairs in formulary")
        else:
            log("DDInter unavailable (network/cert) — keeping curated + ONC only")
    merged = grounding.merge_interactions(curated, ddinter, db)
    n_onc = sum(1 for a in merged.values() for i in a.values() if i.get("source") == "onc")
    db["interactions"] = merged
    log(f"interactions: {n_curated} curated + {sum(len(v) for v in ddinter.values())} DDInter -> "
        f"{sum(len(v) for v in merged.values())} total ({n_onc} forced major by ONC)")
    return db


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip network; apply ONC overrides only")
    ap.add_argument("--no-formulary", action="store_true")
    ap.add_argument("--no-interactions", action="store_true")
    args = ap.parse_args()

    path = config.DRUG_INTERACTIONS_JSON
    with open(path, "r", encoding="utf-8") as fh:
        db = json.load(fh)

    if not args.no_formulary and not args.offline:
        ground_formulary(db)
    if not args.no_interactions:
        ground_interactions(db, offline=args.offline)

    db.setdefault("_meta", {})
    db["_meta"]["grounded_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    db["_meta"]["sources"] = {
        "RxNorm / RxClass (NLM)": "free, no licence — drug class + ATC",
        "openFDA drug label API": "CC0 public domain — monograph, contraindications, DailyMed URL",
        "DDInter 2.0 (SCBDD)": "CC BY-NC — pairwise interaction severities (NON-COMMERCIAL)",
        "ONC High-Priority DDI / CredibleMeds": "expert consensus — forced 'major' overrides",
    }
    db["_meta"]["attribution"] = ("Contains data from RxNorm/RxClass (NLM), openFDA (FDA, CC0), "
                                  "and DDInter 2.0 (CC BY-NC; non-commercial use).")

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(db, fh, indent=2)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
