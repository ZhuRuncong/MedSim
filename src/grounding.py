"""Real-world grounding for the drug reference data.

Sources (licences verified July 2026):
  - RxNorm + RxClass via the RxNav REST API (NLM) — free, no licence needed.
  - openFDA drug-label API (FDA SPL) — CC0 public domain.
  - DDInter 2.0 (SCBDD) — CC BY-NC; pairwise interactions with Major/Moderate/Minor.
  - ONC High-Priority / CredibleMeds — expert-consensus "always alert" pairs.

Pure transforms are separated from the network fetchers so the parsing/merge
logic is unit-testable offline. Fetchers degrade gracefully (return None/{}) so a
build never hard-fails on a network hiccup, and the citation check treats a bot
block (403) as "resolves" to avoid false "fabricated DOI" flags.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

RXNAV = "https://rxnav.nlm.nih.gov/REST"
OPENFDA = "https://api.fda.gov/drug/label.json"
DDINTER_URL = "http://ddinter.scbdd.com/static/media/download/ddinter_downloads_code_{code}.csv"

# --------------------------------------------------------------------------- #
# ATC prefix -> allergy family (most specific first). allergy_class is the ONLY
# safety-critical field, so this map is deliberately conservative.
# --------------------------------------------------------------------------- #
_ATC_ALLERGY = [
    ("J01CA", "penicillin"), ("J01CE", "penicillin"), ("J01CF", "penicillin"), ("J01CR", "penicillin"),
    ("J01DB", "cephalosporin"), ("J01DC", "cephalosporin"), ("J01DD", "cephalosporin"), ("J01DE", "cephalosporin"),
    ("J01EA", "sulfa"), ("J01EB", "sulfa"), ("J01EC", "sulfa"), ("J01ED", "sulfa"), ("J01EE", "sulfa"),
    ("J01FA", "macrolide"), ("J01FF", "lincosamide"),
    ("J01AA", "tetracycline"),
    ("J01MA", "fluoroquinolone"), ("J01MB", "fluoroquinolone"),
    ("J01GB", "aminoglycoside"),
    ("M01A", "nsaid"), ("N02BA", "nsaid"),
    ("N02A", "opioid"),
]

# ONC High-Priority / CredibleMeds high-severity rules, keyed to our formulary.
# A matcher is a drug name, "family:<allergy_class>", or "class:<substring>".
ONC_MAJOR = [
    ("Nitroglycerin", "Sildenafil"),
    ("family:nsaid", "Warfarin"),
    ("Warfarin", "Metronidazole"),
    ("Warfarin", "Trimethoprim-Sulfamethoxazole"),
    ("Warfarin", "Ciprofloxacin"),
    ("Ondansetron", "Azithromycin"),
    ("Ondansetron", "Levofloxacin"),
    ("Ondansetron", "Ciprofloxacin"),
    ("Lisinopril", "Losartan"),
    ("Enalapril", "Losartan"),
    ("Lisinopril", "Potassium Chloride"),
    ("Methotrexate", "Trimethoprim-Sulfamethoxazole"),
    ("family:nsaid", "Methotrexate"),
]

# DDInter keys on generic component names; map our combo/brand names to components.
_DDINTER_ALIAS = {
    "Trimethoprim-Sulfamethoxazole": ["sulfamethoxazole", "trimethoprim", "co-trimoxazole"],
    "Piperacillin-Tazobactam": ["piperacillin", "tazobactam"],
    "Amoxicillin-Clavulanate": ["amoxicillin", "clavulanate"],
    "Insulin (regular)": ["insulin regular", "insulin human", "insulin"],
}


# --------------------------------------------------------------------------- #
# Pure transforms
# --------------------------------------------------------------------------- #
def atc_to_allergy_family(atc: Optional[str]) -> Optional[str]:
    if not atc:
        return None
    a = atc.upper()
    for prefix, fam in _ATC_ALLERGY:
        if a.startswith(prefix):
            return fam
    return None


def _clean(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    text = re.sub(r"^\d+(\.\d+)*\s+", "", text)                      # drop "7 DRUG INTERACTIONS" headers
    text = re.sub(r"^[A-Z][A-Z \-/]{6,}:?\s+", "", text)            # drop ALL-CAPS section headers
    if len(text) > limit:
        cut = text[:limit]
        m = re.search(r"^(.*?[.!?])\s", cut)
        text = (m.group(1) if m else cut).strip()
    return text


def parse_rxclass(data: dict) -> dict:
    """Extract EPC class name + ATC ids from a RxClass response.

    Returns all ATC ids (``atcs``) so the caller can disambiguate — a single
    ingredient (e.g. ibuprofen) appears under several ATCs because of combination
    products, so blindly taking the 'most specific' one can grab an opioid-combo
    class. ``atc`` is the most-specific single id for convenience.
    """
    items = (data.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo") or []
    epc, atcs = None, []
    for it in items:
        c = it.get("rxclassMinConceptItem") or {}
        ct = c.get("classType")
        if ct == "EPC" and not epc:
            epc = c.get("className")
        if ct == "ATC1-4":
            cid = c.get("classId", "")
            if cid and cid not in atcs:
                atcs.append(cid)
    atc = max(atcs, key=len) if atcs else None
    return {"class": epc, "atc": atc, "atcs": atcs}


def _first(v):
    if isinstance(v, list):
        return v[0] if v else ""
    return v or ""


def parse_openfda_label(data: dict) -> dict:
    res = data.get("results") or []
    if not res:
        return {}
    r = res[0]
    openfda = r.get("openfda") or {}
    setid = _first(openfda.get("spl_set_id"))
    url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}" if setid else None
    out = {
        "monograph": _clean(_first(r.get("indications_and_usage"))),
        "contraindications": _clean(_first(r.get("contraindications")), 300),
        "label_url": url,
        "rxcui": _first(openfda.get("rxcui")) or None,
        "epc": _first(openfda.get("pharm_class_epc")) or None,
    }
    bw = _clean(_first(r.get("boxed_warning")), 300)
    if bw:
        out["boxed_warning"] = bw
    return {k: v for k, v in out.items() if v}


def _norm(name: str) -> str:
    n = (name or "").lower().strip()
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"[^a-z0-9\- ]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def ddinter_name_map(formulary_names: List[str]) -> Dict[str, str]:
    """{normalized DDInter drug name -> our formulary name}."""
    m: Dict[str, str] = {}
    for name in formulary_names:
        m[_norm(name)] = name
        for alias in _DDINTER_ALIAS.get(name, []):
            m[_norm(alias)] = name
    return m


def parse_ddinter(rows: List[dict], name_map: Dict[str, str]) -> Dict[str, Dict[str, dict]]:
    """DDInter CSV rows -> directional interaction dict for our formulary."""
    inter: Dict[str, Dict[str, dict]] = {}
    for row in rows:
        a = name_map.get(_norm(row.get("Drug_A", "")))
        b = name_map.get(_norm(row.get("Drug_B", "")))
        lvl = (row.get("Level") or "").strip().lower()
        if a and b and a != b and lvl in ("major", "moderate", "minor"):
            inter.setdefault(a, {})[b] = {
                "severity": lvl,
                "description": f"{row.get('Drug_A')}–{row.get('Drug_B')} interaction ({lvl}, DDInter).",
                "source": "ddinter",
            }
    return inter


def _members(matcher: str, drug_db: dict) -> List[str]:
    drugs = drug_db.get("drugs", {})
    if matcher.startswith("family:"):
        fam = matcher.split(":", 1)[1]
        return [n for n, m in drugs.items() if m.get("allergy_class") == fam]
    if matcher.startswith("class:"):
        sub = matcher.split(":", 1)[1].lower()
        return [n for n, m in drugs.items() if sub in (m.get("class", "") or "").lower()]
    return [matcher] if matcher in drugs else []


def apply_onc_overrides(interactions: dict, drug_db: dict, rules=ONC_MAJOR) -> dict:
    """Force severity=major on expert-consensus high-priority pairs."""
    for ma, mb in rules:
        for a in _members(ma, drug_db):
            for b in _members(mb, drug_db):
                if a == b:
                    continue
                interactions.setdefault(a, {})[b] = {
                    "severity": "major",
                    "description": "High-priority interaction (ONC/CredibleMeds consensus).",
                    "source": "onc",
                }
    return interactions


def merge_interactions(curated: dict, ddinter: dict, drug_db: dict) -> dict:
    """Precedence: ONC-major > curated > DDInter. Curated descriptions survive;
    ONC forces major on top for safety-critical pairs."""
    out: Dict[str, Dict[str, dict]] = {}
    for src in (ddinter, curated):                          # ddinter base, curated overlays
        for a, partners in src.items():
            for b, info in partners.items():
                out.setdefault(a, {})[b] = dict(info)
    apply_onc_overrides(out, drug_db)                       # ONC overlays last
    return out


# --------------------------------------------------------------------------- #
# Network fetchers (graceful — never raise)
# --------------------------------------------------------------------------- #
def _get(url, params=None, timeout=10, **kw):
    import requests

    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": "MedSim/1.0 (education)"}, **kw)
    r.raise_for_status()
    return r


def rxclass_for_name(name: str) -> dict:
    try:
        d = _get(f"{RXNAV}/rxclass/class/byDrugName.json", {"drugName": name}).json()
        return parse_rxclass(d)
    except Exception:
        return {}


def openfda_label_for_name(name: str) -> dict:
    try:
        d = _get(OPENFDA, {"search": f'openfda.generic_name:"{name}"', "limit": 1}).json()
        return parse_openfda_label(d)
    except Exception:
        return {}


def download_ddinter() -> List[dict]:
    """Download + concatenate the DDInter shards. Returns [] on any failure."""
    import csv
    import io

    rows: List[dict] = []
    for code in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        try:
            r = _get(DDINTER_URL.format(code=code), timeout=30, verify=False)
            rows.extend(csv.DictReader(io.StringIO(r.text)))
        except Exception:
            continue
    return rows


def citation_resolves(url: str, timeout: int = 8) -> Optional[bool]:
    """True if a citation URL exists, False if it definitively doesn't, None if
    unknown. Bot blocks (401/403/405/429) count as EXISTS — many publishers
    (doi.org, journals) refuse HEAD requests, and treating that as 'fabricated'
    would be a false positive."""
    import requests

    if not url or not url.startswith("http"):
        return None
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout,
                          headers={"User-Agent": "Mozilla/5.0 (MedSim)"})
        if r.status_code == 405:  # HEAD not allowed -> try a light GET
            r = requests.get(url, allow_redirects=True, timeout=timeout, stream=True,
                             headers={"User-Agent": "Mozilla/5.0 (MedSim)"})
        return False if r.status_code in (404, 410) else True
    except requests.exceptions.RequestException:
        return None
