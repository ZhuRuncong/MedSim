"""AI case generator (opt-in) — full LLM-authored cases with a
generate-then-verify harness for correctness.

Pipeline for one case:
  1. GENERATE  — the model authors a disease dict, constrained to the existing
                 controlled vocabularies (drugs, orderable tests, exams) so the
                 scoring "answer key" can only reference things the engine knows.
  2. VALIDATE  — schema + structural/integrity checks (same rules the test
                 suite enforces on the curated catalog); auto-repair the few
                 mechanical issues (e.g. deviated labs must be orderable).
  3. VERIFY    — N independent adversarial clinical reviewers (fresh context,
                 temperature 0) judge the medical claims. Majority + confidence
                 threshold required to accept; otherwise the errors are fed back
                 and generation is retried.
  4. REGISTER  — accepted cases are tagged with provenance ("AI-generated,
                 unreviewed"), registered in the catalog, and cached to disk.

Everything is provider-agnostic (see ``llm.py``) and fully testable with an
injected ``MockClient``.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from . import config, data_loader, grounding
from .llm import BaseClient, LLMUnavailable, get_client
from .util import stable_seed

VALID_FREQ = {"common", "occasional", "rare"}


class CaseGenerationError(RuntimeError):
    """Generation failed validation/verification after all attempts."""


# --------------------------------------------------------------------------- #
# Output schema (structural only — medical correctness is checked separately)
# --------------------------------------------------------------------------- #
class _Symptom(BaseModel):
    name: str
    frequency: str


class _Guideline(BaseModel):
    name: str
    url: str


class _Demographics(BaseModel):
    age_range: List[int] = Field(min_length=2, max_length=2)
    sex: str = "any"


class _Interaction(BaseModel):
    """A drug-drug interaction the model declares for a new drug."""

    model_config = {"populate_by_name": True}

    partner: str = Field(default="", alias="with")   # the other drug
    severity: str = "moderate"                        # minor | moderate | major
    description: str = ""


class _NewDrug(BaseModel):
    """A drug the model introduces because it isn't in the base formulary."""

    name: str
    drug_class: str = ""
    allergy_class: str = "none"   # must map to a known allergy family or 'none'
    route: str = ""
    monograph: str = ""
    interactions: List[_Interaction] = Field(default_factory=list)


class _Gate(BaseModel):
    """An ordered safety dependency (executable)."""

    type: str = ""                # test_before_drug | test_before_test
    test: str = ""                # for test_before_drug
    drug: str = ""                # for test_before_drug
    first: str = ""               # for test_before_test
    then: str = ""                # for test_before_test
    rationale: str = ""
    penalty: int = 10


class GeneratedDisease(BaseModel):
    name: str
    aliases: List[str] = Field(default_factory=list)
    specialties: List[str] = Field(default_factory=list)
    difficulty: int = 2
    chief_complaint: str
    demographics: _Demographics
    vitals: Dict[str, float] = Field(default_factory=dict)
    symptoms: List[_Symptom] = Field(default_factory=list)
    history: Dict[str, str] = Field(default_factory=dict)
    exam_findings: Dict[str, str] = Field(default_factory=dict)
    appropriate_tests: List[str] = Field(default_factory=list)
    appropriate_exams: List[str] = Field(default_factory=list)
    lab_deviations: Dict[str, dict] = Field(default_factory=dict)
    first_line_drugs: List[str] = Field(default_factory=list)
    reasonable_drugs: List[str] = Field(default_factory=list)
    contraindicated_drugs: List[str] = Field(default_factory=list)
    new_drugs: List[_NewDrug] = Field(default_factory=list)  # self-extending formulary
    indicated_surgeries: List[str] = Field(default_factory=list)
    severity: int = 2                                       # 1 routine … 3 life-threatening
    safety_gates: List[_Gate] = Field(default_factory=list)  # ordered dependencies
    differentials: List[str] = Field(default_factory=list)
    teaching: str = ""
    guideline: _Guideline


# --------------------------------------------------------------------------- #
# Structural / integrity validation (mirrors tests/unit/test_data.py rules)
# --------------------------------------------------------------------------- #
def validate_disease(d: dict) -> List[str]:
    """Return a list of integrity problems (empty ⇒ structurally valid)."""
    problems: List[str] = []
    drugs = set(data_loader.drug_db()["drugs"])
    orderable = set(data_loader.orderable_tests())
    components = data_loader.all_components()
    qualitative = set(data_loader.QUALITATIVE_TESTS)
    exams = set(config.PHYSICAL_EXAMS)
    specialties = set(config.SPECIALTIES)

    if not d.get("name"):
        problems.append("missing name")
    if not d.get("aliases"):
        problems.append("aliases must be non-empty (used for diagnosis matching)")
    specs = d.get("specialties", [])
    if not specs:
        problems.append("specialties must be non-empty")
    for s in specs:
        if s not in specialties:
            problems.append(f"unknown specialty {s!r}")

    syms = d.get("symptoms", [])
    if len(syms) < 3:
        problems.append("need at least 3 symptoms")
    for s in syms:
        if s.get("frequency") not in VALID_FREQ:
            problems.append(f"symptom {s.get('name')!r} has bad frequency {s.get('frequency')!r}")

    if not d.get("vitals"):
        problems.append("vitals must be provided")

    fl = d.get("first_line_drugs", [])
    if not fl:
        problems.append("first_line_drugs must be non-empty")
    for key in ("first_line_drugs", "reasonable_drugs", "contraindicated_drugs"):
        for name in d.get(key, []):
            if name not in drugs:
                problems.append(f"{key}: unknown drug {name!r} (must be in the drug DB)")
    overlap = set(fl) & set(d.get("contraindicated_drugs", []))
    if overlap:
        problems.append(f"drugs both first-line AND contraindicated: {sorted(overlap)}")

    # appropriate_tests MAY be empty — some conditions (e.g. eczema) are purely
    # clinical diagnoses where ordering labs is legitimately low-value.
    for t in d.get("appropriate_tests", []):
        if t not in orderable:
            problems.append(f"appropriate_test {t!r} is not orderable")

    for ex in d.get("exam_findings", {}):
        if ex not in exams:
            problems.append(f"exam_findings key {ex!r} not in PHYSICAL_EXAMS")
    for ex in d.get("appropriate_exams", []):
        if ex not in exams:
            problems.append(f"appropriate_exam {ex!r} not in PHYSICAL_EXAMS")

    for k in d.get("lab_deviations", {}):
        if k not in components and k not in qualitative:
            problems.append(f"lab_deviation key {k!r} is neither a lab component nor a qualitative test")

    if not d.get("differentials"):
        problems.append("differentials must be non-empty")

    g = d.get("guideline", {})
    if not (g.get("url", "").startswith("http")):
        problems.append("guideline.url must be a real http(s) URL")

    ar = d.get("demographics", {}).get("age_range", [])
    if len(ar) != 2 or ar[0] > ar[1]:
        problems.append("demographics.age_range must be [min, max] with min<=max")

    return problems


def _validate_new_drugs(new_drugs) -> tuple:
    """Check AI-authored drug entries. Returns (problems, [(name, meta), ...]).

    The only safety-relevant field is ``allergy_class`` (it drives the allergy
    engine), so it must map to a known family or 'none'. Everything else is
    descriptive. Curated drugs are never overwritten.
    """
    families = data_loader.known_allergy_families() | {"none"}
    existing = set(data_loader.drug_db()["drugs"])
    severities = {"minor", "moderate", "major"}

    # First pass: structurally-valid new drugs (so interactions can reference
    # both existing drugs and other drugs introduced in the same case).
    problems, candidates = [], []
    for nd in new_drugs:
        name = (nd.name or "").strip()
        if not name or name in existing:
            continue
        ac = (nd.allergy_class or "none").strip().lower()
        if ac not in families:
            problems.append(f"new drug {name!r}: allergy_class {ac!r} must be one of "
                            f"{sorted(families)} or 'none'")
            continue
        if not (nd.drug_class or "").strip():
            problems.append(f"new drug {name!r}: missing drug_class")
            continue
        candidates.append((nd, name, ac))

    known = existing | {name for _, name, _ in candidates}

    # Second pass: build metadata, keeping only well-formed interactions that
    # reference a real drug (bad ones are dropped, not fatal — interactions are a
    # bonus layer, unlike the safety-critical allergy_class).
    to_register = []
    for nd, name, ac in candidates:
        inter = {}
        for it in (nd.interactions or []):
            partner = (it.partner or "").strip()
            sev = (it.severity or "").strip().lower()
            if partner and partner != name and partner in known and sev in severities:
                inter[partner] = {"severity": sev, "description": (it.description or "").strip()}
        meta = {"class": nd.drug_class.strip(), "allergy_class": ac,
                "route": (nd.route or "-").strip(), "monograph": (nd.monograph or "").strip(),
                "generated": True}
        if inter:
            meta["interactions"] = inter
        to_register.append((name, meta))
    return problems, to_register


def _auto_repair(d: dict) -> dict:
    """Coerce the *presentation* layer to the engine's controlled vocabularies.

    Labs / tests / exams / specialties are flavour, not the graded answer key, so
    an out-of-vocab entry is dropped (keeping valid ones) rather than wasting a
    generation attempt. Drug lists are filtered too — a partly-unknown list keeps
    its valid members (e.g. drop 'Colchicine', keep 'Ibuprofen'); validation only
    fails if NO valid first-line drug remains. The graded answer key is therefore
    always a *subset* of what the model proposed, never something invented.
    """
    drugs = set(data_loader.drug_db()["drugs"])
    orderable = set(data_loader.orderable_tests())
    components = data_loader.all_components()
    qualitative = set(data_loader.QUALITATIVE_TESTS)
    exams = set(config.PHYSICAL_EXAMS)

    for key in ("first_line_drugs", "reasonable_drugs", "contraindicated_drugs"):
        d[key] = [x for x in d.get(key, []) if x in drugs]
    # A drug can't be both first-line and contraindicated.
    fl = set(d["first_line_drugs"])
    d["contraindicated_drugs"] = [x for x in d["contraindicated_drugs"] if x not in fl]

    d["lab_deviations"] = {k: v for k, v in d.get("lab_deviations", {}).items()
                           if k in components or k in qualitative}
    d["exam_findings"] = {k: v for k, v in d.get("exam_findings", {}).items() if k in exams}
    d["appropriate_exams"] = [e for e in d.get("appropriate_exams", []) if e in exams]
    for ex in d["exam_findings"]:
        if ex not in d["appropriate_exams"]:
            d["appropriate_exams"].append(ex)

    tests = [t for t in d.get("appropriate_tests", []) if t in orderable]
    for k in d["lab_deviations"]:  # any lab we deviate must be orderable & appropriate
        panel = data_loader.panel_for_component(k) or (k if k in qualitative else None)
        if panel and panel not in tests:
            tests.append(panel)
    d["appropriate_tests"] = tests

    # Clamp severity; keep only executable safety gates referencing known vocab.
    try:
        d["severity"] = min(3, max(1, int(d.get("severity", 2))))
    except (TypeError, ValueError):
        d["severity"] = 2
    good_gates = []
    for g in d.get("safety_gates", []) or []:
        t = g.get("type")
        if t == "test_before_drug" and g.get("test") in orderable and g.get("drug") in drugs:
            good_gates.append(g)
        elif t == "test_before_test" and g.get("first") in orderable and g.get("then") in orderable \
                and g.get("first") != g.get("then"):
            good_gates.append(g)
    for i, g in enumerate(good_gates):
        g["id"] = f"gate:{i}"
        # a gate's prerequisite test should itself be part of the appropriate work-up
        pre = g.get("test") or g.get("first")
        if pre and pre not in d["appropriate_tests"]:
            d["appropriate_tests"].append(pre)
    d["safety_gates"] = good_gates
    return d


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def _vocab_block() -> str:
    drugs = ", ".join(data_loader.all_drug_names())
    tests = ", ".join(data_loader.orderable_tests())
    exams = ", ".join(config.PHYSICAL_EXAMS)
    components = ", ".join(sorted(data_loader.all_components()))
    return (
        f"ALLOWED DRUGS (first_line_drugs / reasonable_drugs / contraindicated_drugs "
        f"must be chosen ONLY from this list):\n{drugs}\n\n"
        f"ALLOWED TESTS (appropriate_tests must be from this list):\n{tests}\n\n"
        f"ALLOWED PHYSICAL EXAMS (exam_findings keys must be from this list):\n{exams}\n\n"
        f"LAB COMPONENTS you may use as numeric lab_deviations keys (each maps to a panel):\n{components}\n\n"
        f"For qualitative/imaging lab_deviations, the key must be one of the ALLOWED TESTS above "
        f"and the value must be {{\"finding\": \"...\"}}. For numeric components use {{\"value\": <number>}}."
    )


_GEN_SYSTEM = (
    "You are a medical education content author creating a realistic, single-diagnosis "
    "training case for a clinical simulator. Output must be evidence-based and mapped to a "
    "real clinical practice guideline (with a genuine URL/DOI). You MUST only reference drugs, "
    "tests and exams from the provided controlled vocabularies. Return a single JSON object "
    "matching the requested schema exactly."
)


# Inline one-shot FORMAT exemplar (prompt scaffolding — NOT a playable catalog
# case; the app registers no hard-coded cases). It only shows the model the
# expected JSON shape using in-vocabulary drugs/tests/exams.
_EXAMPLE_CASE = {
    "name": "Acute Cystitis",
    "aliases": ["cystitis", "lower urinary tract infection", "bladder infection", "uti"],
    "specialties": ["Internal Medicine"],
    "difficulty": 1,
    "chief_complaint": "Burning with urination and urinary frequency for 2 days",
    "demographics": {"age_range": [20, 45], "sex": "female"},
    "vitals": {"HR": 82, "SBP": 118, "DBP": 74, "RR": 16, "Temp": 37.2, "SpO2": 99},
    "symptoms": [
        {"name": "dysuria", "frequency": "common"},
        {"name": "urinary frequency and urgency", "frequency": "common"},
        {"name": "suprapubic discomfort", "frequency": "occasional"},
    ],
    "history": {
        "onset": "It started two days ago and keeps getting worse.",
        "urination": "It burns when I go and I need to go constantly.",
        "fever": "No fever or back pain.",
    },
    "exam_findings": {"Palpate abdomen": "Mild suprapubic tenderness; no flank tenderness."},
    "appropriate_tests": ["Urinalysis", "Urine Culture"],
    "appropriate_exams": ["Palpate abdomen"],
    "lab_deviations": {
        "Urinalysis": {"finding": "Positive leukocyte esterase and nitrites; WBCs and bacteria."},
        "Urine Culture": {"finding": "Pending; >100,000 CFU/mL gram-negative rods."},
    },
    "first_line_drugs": ["Nitrofurantoin", "Trimethoprim-Sulfamethoxazole"],
    "reasonable_drugs": ["Ciprofloxacin"],
    "contraindicated_drugs": [],
    "new_drugs": [],
    "indicated_surgeries": [],
    "severity": 1,
    "safety_gates": [],
    "differentials": ["Pyelonephritis", "Vaginitis", "Urethritis"],
    "teaching": "Uncomplicated cystitis is treated empirically; nitrofurantoin or TMP-SMX are first-line.",
    "guideline": {"name": "IDSA Uncomplicated Cystitis Guideline 2011",
                  "url": "https://doi.org/10.1093/cid/ciq257"},
}


def _example_case() -> dict:
    return dict(_EXAMPLE_CASE)


def _gen_user(specialty: str, difficulty: Optional[int],
              avoid: List[str], errors: List[str]) -> str:
    diff = difficulty or 2
    history_keys = ("onset, chief-complaint detail, associated symptoms, risk factors, "
                    "past medical history, medications, allergies, social history")
    parts = [
        f"Create ONE {specialty} case at difficulty {diff} (1=easy … 3=hard).",
        f"The 'specialties' field MUST contain only values from this exact list, and "
        f"MUST include \"{specialty}\": {config.SPECIALTIES}.",
        f"Pick a DISTINCT condition NOT in this list: {', '.join(sorted(avoid))}.",
        "",
        _vocab_block(),
        "",
        "Requirements:",
        "- Prefer drugs already in the ALLOWED DRUGS list. If the guideline first-line therapy "
        "needs a drug NOT in that list, INTRODUCE it: add an entry to 'new_drugs' as "
        "{name, drug_class, allergy_class, route, monograph, interactions}, then reference that "
        "exact name in first_line_drugs/contraindicated_drugs. Do NOT substitute an inferior "
        "in-list drug just to avoid this (a reviewer will reject a wrong first-line choice).",
        f"- 'new_drugs[].allergy_class' MUST be one of {sorted(data_loader.known_allergy_families())} "
        "or 'none' (this drives allergy safety-checking).",
        "- 'new_drugs[].interactions' lists important drug-drug interactions as "
        "[{with, severity, description}], where 'with' is another drug name (an ALLOWED drug or "
        "another new drug) and severity is minor|moderate|major (e.g. a new anticoagulant "
        "interacts 'major' with Aspirin). Use [] if none.",
        "- 'first_line_drugs' MUST be the guideline first-line treatment(s) for this exact condition.",
        "- 'contraindicated_drugs' MUST be genuinely harmful/contraindicated in this condition "
        "(not merely 'not indicated'); [] if none apply.",
        "- 'appropriate_tests' are the tests a clinician would actually order in the work-up.",
        "- 'lab_deviations' encode this disease's characteristic abnormal results.",
        "- IMPORTANT: 'exam_findings' and 'lab_deviations' findings must describe the objective "
        "sign or result ONLY and must NOT name the diagnosis or its abbreviation (the student is "
        "still working it out). Write 'ST-segment elevation in the inferior leads', not "
        "'ECG consistent with STEMI'; 'right lower quadrant guarding', not 'signs of appendicitis'.",
        "- 'symptoms' each need a frequency of common | occasional | rare.",
        f"- 'history' is a dict of short first-person patient answers keyed by topic ({history_keys}).",
        "- 'exam_findings' maps an allowed exam name to what is found.",
        "- 'differentials' are plausible alternative diagnoses.",
        "- 'severity' is the clinical stakes: 1 = routine/self-limited, 2 = urgent, "
        "3 = life-threatening if missed or mismanaged.",
        "- 'safety_gates' encode ORDERED safety dependencies (executed by the engine). Use "
        "{type:'test_before_drug', test, drug, rationale, penalty} when a lab must precede a drug "
        "(e.g. check 'BMP' before 'Insulin (regular)' in DKA — potassium), and "
        "{type:'test_before_test', first, then, rationale, penalty} when one test must precede another "
        "(e.g. 'Beta-hCG' before 'CT Abdomen/Pelvis' in a person of child-bearing potential). "
        "Reference ONLY allowed tests/drugs; penalty is a positive int; use [] if none apply.",
        "- 'guideline' must cite a real society/guideline with a working URL or DOI.",
        "- 'aliases' include common lay terms and abbreviations for the diagnosis.",
        "",
        "Return JSON with EXACTLY these keys: name, aliases, specialties, difficulty, "
        "chief_complaint, demographics{age_range:[min,max], sex}, vitals{HR,SBP,DBP,RR,Temp,SpO2}, "
        "symptoms[{name,frequency}], history{}, exam_findings{}, appropriate_tests[], "
        "appropriate_exams[], lab_deviations{}, first_line_drugs[], reasonable_drugs[], "
        "contraindicated_drugs[], "
        "new_drugs[{name,drug_class,allergy_class,route,monograph,interactions:[{with,severity,description}]}], "
        "indicated_surgeries[], severity, "
        "safety_gates[{type,test,drug,first,then,rationale,penalty}], "
        "differentials[], teaching, guideline{name,url}.",
        "",
        "FORMAT EXAMPLE (structure only — invent a different condition):",
        json.dumps(_example_case(), indent=2),
    ]
    if errors:
        parts += ["", "Your previous attempt was rejected. FIX THESE ISSUES:",
                  *[f"- {e}" for e in errors]]
    return "\n".join(parts)


_VERIFY_SYSTEM = (
    "You are a board-certified physician doing a safety review of an AI-generated teaching case. "
    "Judge CORRECTNESS, not completeness. A case PASSES when everything listed is medically correct "
    "for the diagnosis: the listed first-line drugs are genuinely first-line, the contraindicated "
    "drugs are genuinely contraindicated, the tests are appropriate, and the guideline is real and "
    "relevant. Do NOT reject merely because a list is non-exhaustive, an additional option could be "
    "added, or you would have phrased it differently — a correct SUBSET is acceptable. "
    "REJECT only for real errors: a wrong or harmful 'first-line' drug, a false contraindication, an "
    "inappropriate test, an incorrect diagnosis, or a fabricated guideline. Respond as one JSON "
    'object: {"status": "pass"|"revise"|"reject", "confidence": 0.0-1.0, "issues": [strings]}.'
)


def _verify_user(d: dict) -> str:
    claims = {
        "name": d.get("name"),
        "chief_complaint": d.get("chief_complaint"),
        "key_symptoms": [s["name"] for s in d.get("symptoms", [])],
        "first_line_drugs": d.get("first_line_drugs"),
        "reasonable_drugs": d.get("reasonable_drugs"),
        "contraindicated_drugs": d.get("contraindicated_drugs"),
        "appropriate_tests": d.get("appropriate_tests"),
        "indicated_surgeries": d.get("indicated_surgeries"),
        "severity_1_to_3": d.get("severity"),
        "safety_gates": d.get("safety_gates"),
        "differentials": d.get("differentials"),
        "guideline": d.get("guideline"),
    }
    # Include the identity (class + allergy family) of every referenced drug so the
    # reviewer can catch a mis-identified or newly-introduced ('generated') drug.
    referenced = set(d.get("first_line_drugs", []) + d.get("reasonable_drugs", [])
                     + d.get("contraindicated_drugs", []))
    details = {}
    for name in referenced:
        meta = data_loader.drug_info(name) or {}
        entry = {"class": meta.get("class"), "allergy_class": meta.get("allergy_class"),
                 "ai_authored": bool(meta.get("generated"))}
        if meta.get("contraindications"):   # real FDA-label text (grounded)
            entry["fda_contraindications"] = meta["contraindications"][:200]
        if meta.get("generated") and meta.get("interactions"):
            entry["interactions"] = {p: i.get("severity") for p, i in meta["interactions"].items()}
        details[name] = entry
    claims["drug_details"] = details
    return ("Review this case's medical answer key. For the stated diagnosis, are the first-line "
            "drugs truly guideline first-line? Are the contraindicated drugs truly contraindicated? "
            "Are the appropriate tests appropriate? Is the guideline real and relevant? Check that "
            "each drug in 'drug_details' is real and correctly classified (especially any marked "
            "ai_authored), and that any interactions listed for them are clinically plausible. "
            "List concrete issues.\n\n" + json.dumps(claims, indent=2))


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_case(disease: dict, client: BaseClient,
                n_verifiers: int, min_confidence: float) -> dict:
    votes = []
    for _ in range(max(1, n_verifiers)):
        try:
            votes.append(client.complete_json(_VERIFY_SYSTEM, _verify_user(disease), temperature=0.0))
        except Exception as e:  # a failed reviewer counts as a reject
            votes.append({"status": "reject", "confidence": 0.0, "issues": [f"verifier error: {e}"]})

    statuses = [str(v.get("status", "reject")).lower() for v in votes]
    confs = [float(v.get("confidence", 0.0) or 0.0) for v in votes]
    issues = [i for v in votes for i in v.get("issues", [])]
    mean_conf = sum(confs) / len(confs)
    accepted = (
        "reject" not in statuses
        and statuses.count("pass") >= (len(statuses) + 1) // 2
        and mean_conf >= min_confidence
    )
    return {"accepted": accepted, "issues": issues, "confidence": round(mean_conf, 3),
            "statuses": statuses}


# --------------------------------------------------------------------------- #
# Conversion + top-level generate
# --------------------------------------------------------------------------- #
def _ground_citation(disease: dict):
    """Check the guideline URL resolves. Returns True (resolves), None (unknown /
    network down), or False (definitively 404s) — and in the False case replaces
    the fabricated URL with a real PubMed search link, keeping the original for
    transparency. Never raises; never rejects the case."""
    g = disease.get("guideline") or {}
    url = g.get("url")
    resolves = grounding.citation_resolves(url)
    if resolves is False:
        g["unverified_url"] = url
        term = (disease.get("name", "") + " guideline").strip().replace(" ", "+")
        g["url"] = "https://pubmed.ncbi.nlm.nih.gov/?term=" + term
        disease["guideline"] = g
    return resolves


def _to_disease_dict(model: GeneratedDisease, specialty: str) -> dict:
    d = model.model_dump()
    # Specialty labels are routing metadata, not a medical claim — keep only
    # valid ones and guarantee the requested specialty is present (auto-repair
    # rather than reject, so an out-of-vocab label doesn't waste an attempt).
    allowed = set(config.SPECIALTIES)
    specs = [s for s in d.get("specialties", []) if s in allowed]
    if specialty not in specs:
        specs.insert(0, specialty)
    d["specialties"] = specs
    d["id"] = f"GEN-{stable_seed(model.name):08x}"
    d["generated"] = True
    return _auto_repair(d)


def generate_case(
    specialty: str,
    difficulty: Optional[int] = None,
    client: Optional[BaseClient] = None,
    *,
    n_verifiers: Optional[int] = None,
    max_attempts: Optional[int] = None,
    min_confidence: Optional[float] = None,
    avoid_names: Optional[List[str]] = None,
    register: bool = True,
) -> dict:
    """Generate, validate and verify one new case for ``specialty``.

    Raises :class:`LLMUnavailable` if no client/provider is configured, or
    :class:`CaseGenerationError` if no attempt passes validation + verification.
    """
    client = client or get_client()
    if client is None:
        raise LLMUnavailable("No LLM provider configured. Set GOOGLE_API_KEY or ANTHROPIC_API_KEY.")

    n_verifiers = config.GEN_VERIFIERS if n_verifiers is None else n_verifiers
    max_attempts = config.GEN_MAX_ATTEMPTS if max_attempts is None else max_attempts
    min_confidence = config.GEN_MIN_CONFIDENCE if min_confidence is None else min_confidence
    avoid = avoid_names or [d["name"] for d in data_loader.diseases()]

    errors: List[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            raw = client.complete_json(_GEN_SYSTEM, _gen_user(specialty, difficulty, avoid, errors),
                                       temperature=0.4)
        except Exception as e:
            errors = [f"generation call failed: {e}"]
            continue

        try:
            model = GeneratedDisease.model_validate(raw)
        except ValidationError as e:
            errors = [f"schema error: {err['loc']}: {err['msg']}" for err in e.errors()[:8]]
            continue

        # Self-extending formulary: validate & register any AI-authored drugs
        # BEFORE the case is validated, so first-line references resolve.
        drug_problems, new_drugs = _validate_new_drugs(model.new_drugs)
        if drug_problems:
            errors = drug_problems
            continue
        for name, meta in new_drugs:
            data_loader.register_drug(name, meta)

        disease = _to_disease_dict(model, specialty)
        problems = validate_disease(disease)
        if problems:
            errors = problems
            continue

        verdict = verify_case(disease, client, n_verifiers, min_confidence)
        if not verdict["accepted"]:
            errors = verdict["issues"] or ["verification did not reach acceptance threshold"]
            continue

        # Grounding: don't PRESENT a fabricated citation as real. LLMs routinely
        # invent plausible guideline *page* URLs that 404, so rather than fail the
        # whole (medically-verified) case, we replace a non-resolving URL with a
        # real literature-search link and flag it as unverified.
        citation_verified = _ground_citation(disease) if config.CHECK_CITATIONS else None

        # Which of the new drugs actually ended up used in the accepted case.
        used = set(disease.get("first_line_drugs", []) + disease.get("reasonable_drugs", [])
                   + disease.get("contraindicated_drugs", []))
        added = [(n, m) for n, m in new_drugs if n in used]
        disease["provenance"] = {
            "source": "ai-generated",
            "reviewed_by_human": False,
            "model": client.name,
            "attempts": attempt,
            "verification": verdict,
            "new_drugs": [n for n, _ in added],
            "citation_verified": citation_verified,
        }
        if register:
            data_loader.register_disease(disease)
            try:
                data_loader.persist_generated(disease)
                for name, meta in added:
                    data_loader.persist_generated_drug(name, meta)
            except Exception:
                pass  # caching is best-effort
        return disease

    raise CaseGenerationError(
        f"Could not generate a verified {specialty} case after {max_attempts} attempts. "
        f"Last issues: {errors}")
