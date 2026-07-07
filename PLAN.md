\## Full Implementation Roadmap – Gamified Medical‑Student Hospital Simulation  



\*\*Target:\*\* Web‑only (Streamlit) interactive patient case platform for up to 100 concurrent users.  

\*\*Core idea:\*\* A student selects one or more specialties, receives a hidden disease case, asks questions, performs exams, orders tests, prescribes drugs, can request emergency surgery, and finally makes a diagnosis. Points are awarded/penalized \*\*strictly by evidence‑based guidelines\*\*.  



\---  



\### 1️⃣ Project Foundations  



| Item | Action | Outcome |

|------|--------|----------|

| \*\*Repository\*\* | `git init MedSim \&\& cd MedSim` | Empty repo ready for code. |

| \*\*`.gitignore`\*\* | Add `venv/, \_\_pycache\_\_, .env, \*.log, .DS\_Store` | Clean commits. |

| \*\*Virtual‑env\*\* | `python -m venv venv \&\& .\\venv\\Scripts\\activate` (Windows) | Isolated environment. |

| \*\*`requirements.txt`\*\* | ```text\\nstreamlit\\nlangchain\\nlanggraph\\nqdrant-client\\nredis\\npython-dotenv\\nfastapi\\nuvicorn\\nfastmcp\\ntavily\\nsemantic-scholar\\npymed\\npandas\\nnumpy\\n``` | All runtime dependencies. |

| \*\*`.env`\*\* (never commit) | Store API keys for Gemini, Tavily, Qdrant, Redis, LangSmith. | Secure credentials. |

| \*\*Folder layout\*\* | ```text\\nMedSim/\\n├─ src/               # source code\\n│   ├─ agents/        # LangGraph agents\\n│   ├─ tools/         # FastMCP tool wrappers\\n│   ├─ state.py       # shared GameState model\\n│   └─ config.py      # constants / rubric\\n├─ data/               # medical ontologies, symptom tables\\n├─ scripts/            # one‑off setup scripts\\n├─ tests/               # unit \& integration tests\\n├─ app.py               # Streamlit UI\\n└─ README.md           # high‑level docs\\n``` | Clear project structure. |

| \*\*CI pipeline\*\* | GitHub Actions: lint → unit tests → build Docker image → push. | Automated quality check. |



\---  



\### 2️⃣ Data Foundations (Day 1)  



| Sub‑task | Source | Process | Stored As |

|----------|--------|---------|-----------|

| \*\*Disease‑Symptom mapping\*\* | \*\*UMLS / SNOMED‑CT\*\* (download via NLM) | Parse `ConceptRelationship` table → keep only \*Disease → Finding\* edges; attach frequency (common, occasional, rare). | `data/diseases\_symptoms.csv` |

| \*\*Drug‑Interaction matrix\*\* | \*\*DrugBank Open Data\*\* (CSV dump) | Build `{drug\_a:{drug\_b:interaction\_type}}` JSON. | `data/drug\_interactions.json` |

| \*\*Lab reference ranges\*\* | \*\*MIMIC‑IV\*\* labs table (de‑identified) | Extract normal ranges per test, store as CSV. | `data/lab\_refs.csv` |

| \*\*Evidence‑based point rubric\*\* | \*\*Clinical practice guidelines\*\* (e.g., UpToDate, NICE, ACC) + \*\*Cochrane\*\* systematic reviews | Encode each actionable step (order appropriate test, prescribe guideline‑first‑line med, avoid contraindicated drug) → numeric score (+10, –5, etc.). | `src/config.py` (constants) |

| \*\*Embeddings\*\* | Text of disease descriptions \& drug monographs | `sentence‑transformers` → 384‑dim vectors; batch‑upload to \*\*Qdrant\*\* (two collections: `diseases`, `drugs`). | Qdrant cloud (free tier) |

| \*\*Specialty list\*\* | Defined manually (e.g., Internal Medicine, Emergency, Pediatrics, Dermatology, OB‑GYN, Surgery) | Stored in `src/config.py` for UI checkbox. | `src/config.py` |



Create a \*\*bootstrap script\*\* `scripts/initialize\_data.py` that runs all steps, logs counts, and verifies that Qdrant collections exist.



\---  



\### 3️⃣ FastMCP Custom Tools (Day 2)  



All tools exposed as \*\*FastAPI\*\* endpoints behind the FastMCP gateway (single Docker container).  



| Tool | Endpoint (POST) | Input | Core Logic |

|------|----------------|-------|------------|

| `symptom\_generator` | `/api/v1/symptoms` | `{ "disease\_id": "C0011849" }` | Sample symptoms according to the probability table (use `numpy.random.choice` weighted). |

| `lab\_simulator` | `/api/v1/labs` | `{ "ordered\_tests": \["CBC","BMP"], "disease\_id": … }` | Pull normal ranges, add disease‑specific deviations (e.g., ↑ALT for hepatitis). |

| `drug\_effect\_engine` | `/api/v1/drug` | `{ "drug": "Amoxicillin", "patient\_allergies": \[...], "disease\_id": … }` | Check allergy + drug‑interaction matrix; fetch efficacy data from PubMed abstracts (cached). Return `{effect\_score, contraindication\_flag, citations}`. |

| `knowledge\_lookup` | `/api/v1/lookup` | `{ "query": "first‑line treatment for community‑acquired pneumonia?" }` | Look up in local DB (guidelines table). If missing, call \*\*Tavily API\*\*, parse top 3 snippets, summarize; include source URLs. |

| `emergency\_surgery` | `/api/v1/surgery` | `{ "procedure": "Appendectomy", "disease\_id": … }` | Validate that surgery is indicated by guidelines (e.g., perforated appendix); return outcome probability and point impact. |



All responses include \*\*`citations`\*\* (DOI or URL) for transparency.



\---  



\### 4️⃣ Shared Game State (LangGraph)  



File: `src/state.py`  



```python

from pydantic import BaseModel, Field

from typing import List, Dict, Optional



class GameState(BaseModel):

&#x20;   # Core patient data (fixed per case)

&#x20;   case\_id: str

&#x20;   disease\_id: str

&#x20;   specialty: str                     # selected by student

&#x20;   symptoms: List\[str]                # generated at start

&#x20;   vitals: Dict\[str, float]           # HR, BP, RR, Temp, SpO2

&#x20;   # Dynamic interaction history

&#x20;   asked\_history: List\[str] = Field(default\_factory=list)

&#x20;   performed\_exam: List\[str] = Field(default\_factory=list)

&#x20;   ordered\_tests: List\[str] = Field(default\_factory=list)

&#x20;   test\_results: Dict\[str, Dict] = Field(default\_factory=dict)

&#x20;   prescribed\_drugs: List\[str] = Field(default\_factory=list)

&#x20;   surgery\_requested: Optional\[str] = None

&#x20;   diagnosis: Optional\[str] = None

&#x20;   # Scoring \& flow control

&#x20;   points: int = 0

&#x20;   turn: int = 0

&#x20;   retries: int = 0                 # number of invalid actions

&#x20;   max\_retries: int = 3

```



The state is stored in \*\*Redis\*\* (`redis-py`) under key `session:<student\_id>`. All agents read and write atomically (Redis transactions).



\---  



\### 5️⃣ LangGraph Agents (Day 3)  



| Agent | Role | Trigger / Input | Primary Output / Side‑Effect |

|-------|------|----------------|------------------------------|

| \*\*PatientAgent\*\* | Generates intake (age, gender, chief complaint, vitals) \& supplies symptom answers to history questions. | First turn after case creation. | Populates `GameState.symptoms`, `vitals`. |

| \*\*SupervisorAgent\*\* | Central router, applies evidence‑based rubric, updates points, enforces retry limits, decides next allowed actions. | After every student action (question, exam, test, drug, surgery, diagnosis). | Updates `points`, `turn`, possibly sets `retries`. |

| \*\*ClinicalAgent\*\* | Handles \*\*lab ordering\*\* \& \*\*drug prescription\*\*. Calls `lab\_simulator` \& `drug\_effect\_engine`. | Receives ordered test list or drug name from UI. | Updates `test\_results` or `prescribed\_drugs`; awards points per rubric. |

| \*\*SurgeryAgent\*\* | Processes \*\*emergency surgery\*\* requests. Calls `emergency\_surgery`. | Student selects a procedure from UI. | Sets `surgery\_requested`; adjusts points (positive if indicated, negative if not). |

| \*\*KnowledgeAgent\*\* | Answers factual queries (mechanisms, side‑effects, epidemiology). | Student asks a free‑text question. | Returns concise answer with citations; points may be awarded for correct usage. |

| \*\*CriticAgent\*\* | Evaluates final \*\*diagnosis\*\* against hidden disease; awards large bonus if correct, penalties otherwise. | Student submits diagnosis. | Updates `points`; signals end‑of‑case or retry. |



All agents are \*\*stateless functions\*\* that receive the current `GameState` (from Redis) and return an updated state plus any UI messages.



The \*\*LangGraph workflow\*\* (visualised in Mermaid) enforces a turn‑based loop:



```mermaid

graph TD

&#x20;   START\[Create New Case] --> PAT\[PatientAgent]

&#x20;   PAT --> SUP\[SupervisorAgent]

&#x20;   SUP --> |Ask History| PAT

&#x20;   SUP --> |Perform Exam| PAT

&#x20;   SUP --> |Order Test| CLIN\[ClinicalAgent]

&#x20;   SUP --> |Prescribe Drug| CLIN

&#x20;   SUP --> |Request Surgery| SURG\[SurgeryAgent]

&#x20;   SUP --> |Ask Knowledge Q| KNOW\[KnowledgeAgent]

&#x20;   SUP --> |Submit Diagnosis| CRIT\[CriticAgent]

&#x20;   CRIT --> |Correct| END\[Case Complete → Level Up]

&#x20;   CRIT --> |Incorrect| SUP

&#x20;   SUP --> |Max Retries| END

```



\---  



\### 6️⃣ Front‑End UI (Streamlit) – Day 4  



1\. \*\*Sidebar – Student Profile\*\*  

&#x20;  - Name, current level, total points.  

&#x20;  - \*\*Specialty selector\*\* (checkbox list from `src/config.SPECIALTIES`).  

&#x20;  - “Start New Case” button (creates `GameState` with random disease from selected specialties).  



2\. \*\*Main Panel – Patient Card\*\*  

&#x20;  - Vitals (HR, BP, Temp, SpO₂).  

&#x20;  - Chief complaint.  

&#x20;  - “Ask History Question” text box.  

&#x20;  - “Perform Physical Exam” multiselect (e.g., “Inspect skin”, “Auscultate lungs”).  



3\. \*\*Action Panel\*\*  

&#x20;  - \*\*Order Tests\*\* – multiselect of available labs/imaging (filtered by specialty).  

&#x20;  - \*\*Prescribe Medication\*\* – dropdown with auto‑complete (searches drug collection).  

&#x20;  - \*\*Emergency Surgery\*\* – dropdown of procedures (only those indicated for the disease).  

&#x20;  - \*\*Submit Diagnosis\*\* – free‑text field.  



4\. \*\*Log / Feed\*\*  

&#x20;  - Chronological list of student actions, system responses, points delta (green for gain, red for loss).  

&#x20;  - Inline citations as clickable links.  



5\. \*\*Scoring Display\*\*  

&#x20;  - Current point total, turn number, remaining retries.  

&#x20;  - Small animation (confetti) on positive actions, red flash on penalties.  



\*\*Styling\*\* (premium look):  

\- Use \*\*Google Font “Outfit”\*\*.  

\- Dark‑mode background with subtle \*\*glass‑morphism cards\*\* (CSS `backdrop-filter: blur(8px)`); hover transitions for buttons.  

\- Consistent \*\*color palette\*\* (deep teal, amber accent).  



All UI components are wrapped in reusable Streamlit functions (`components/`) for easy theming.



\---  



\### 7️⃣ Evidence‑Based Scoring Rubric (Day 5)  



Located in `src/config.py`:



```python

\# Example rubric (points can be tuned later)

POINTS = {

&#x20;   "order\_appropriate\_test": 10,

&#x20;   "order\_unnecessary\_test": -5,

&#x20;   "prescribe\_guideline\_first\_line": 15,

&#x20;   "prescribe\_contraindicated\_drug": -20,

&#x20;   "avoid\_allergy": 5,

&#x20;   "correct\_diagnosis": 30,

&#x20;   "incorrect\_diagnosis": -15,

&#x20;   "emergency\_surgery\_indicated": 20,

&#x20;   "unwarranted\_surgery": -25,

&#x20;   "use\_knowledge\_query": 2,   # small reward for using evidence source

}

```



Each agent calls `apply\_points(action\_name, state)` → updates `state.points`.  



All point assignments are \*\*fully traceable\*\* in LangSmith (agent name, action, points delta, source guideline URL).



\---  



\### 8️⃣ Testing \& Validation (Day 6)  



| Test | Tool | Goal |

|------|------|------|

| \*\*Unit\*\* | `pytest tests/unit/` | Validate each FastMCP tool returns correct format; verify rubric calculations. |

| \*\*Integration\*\* | `pytest tests/integrations/test\_full\_flow.py` | Simulate a whole case (history → labs → drug → diagnosis) and assert final points match expected guideline outcome. |

| \*\*Ragas\*\* | `ragas evaluate` | Check factual correctness of KnowledgeAgent answers against PubMed citations. |

