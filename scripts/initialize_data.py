"""Reference-data bootstrap / health-check.

MedSim ships NO hard-coded disease cases — every case is authored at runtime by
the AI generator. This script validates the *reference vocabulary* the generator
and simulator are built on (drugs, interactions, allergy classes, lab ranges,
exams) and optionally uploads drug embeddings to Qdrant.

Usage:  python scripts/initialize_data.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data_loader  # noqa: E402


def log(msg: str) -> None:
    print(f"[init] {msg}")


def verify_reference_data() -> int:
    """Cross-check the reference vocabulary; return the number of problems."""
    problems = []
    db = data_loader.drug_db()
    drugs = set(db.get("drugs", {}))

    # Allergy-class members must be real drugs.
    for fam, members in db.get("allergy_classes", {}).items():
        for m in members:
            if m not in drugs:
                problems.append(f"allergy_class {fam!r}: unknown drug {m!r}")
    # Interaction partners must be real drugs.
    for a, partners in db.get("interactions", {}).items():
        for b in partners:
            if a not in drugs or b not in drugs:
                problems.append(f"interaction {a!r}/{b!r}: unknown drug")
    # Lab panels must parse into numeric components.
    for panel in data_loader.numeric_panels():
        if not data_loader.components_for_test(panel):
            problems.append(f"lab panel {panel!r} has no components")

    for p in problems:
        log("PROBLEM: " + p)
    return len(problems)


def maybe_upload_embeddings() -> None:
    if not config.QDRANT_URL:
        log("Qdrant not configured (QDRANT_URL empty) — skipping embeddings.")
        return
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore
    except Exception as e:  # pragma: no cover
        log(f"Embedding libs unavailable ({e}); skipping.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None)
    drugs = [{"name": n, **m} for n, m in data_loader.drug_db()["drugs"].items()]
    client.recreate_collection(
        "drugs", vectors_config=VectorParams(size=384, distance=Distance.COSINE))
    vectors = model.encode([f"{d['name']}: {d.get('monograph','')}" for d in drugs]).tolist()
    client.upsert("drugs", [PointStruct(id=i, vector=v, payload=d)
                            for i, (d, v) in enumerate(zip(drugs, vectors))])
    log(f"Qdrant collection 'drugs': {len(drugs)} points uploaded.")


def main() -> int:
    log("=== MedSim reference-data check ===")
    log(f"Drugs in formulary:     {len(data_loader.all_drug_names())}")
    log(f"Orderable tests:        {len(data_loader.orderable_tests())}")
    log(f"Physical exams:         {len(config.PHYSICAL_EXAMS)}")
    log(f"Specialties:            {len(config.SPECIALTIES)}")
    log("Disease cases:          0 hard-coded — every case is AI-generated at runtime.")

    problems = verify_reference_data()
    log(f"Reference-data check: {problems} problem(s).")

    maybe_upload_embeddings()
    log("Done." if problems == 0 else "Done WITH PROBLEMS.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
