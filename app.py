"""
app.py — FastAPI web server for the resume job matcher + resume chatbot.
Run with: uvicorn app:app --reload
Then open: http://localhost:8000
"""

import os
import json
import shutil
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH         = "data_science_job.csv"
DB_LOCATION      = "./jobs_db"
COLLECTION       = "data_science_jobs"
TOP_K            = 20
SHOW_TOP         = 5
UPLOAD_DIR       = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_RESUME_CHARS = 24000

# ── In-memory resume store (session-based) ────────────────────────────────────
# The browser sends a session_id (timestamp) with every chat message
# so the server knows which resume to reference.
_resume_store: dict = {}

# ── Load locations from CSV at startup ───────────────────────────────────────
_df = pd.read_csv(CSV_PATH, encoding="latin-1")
_df.columns = _df.columns.str.strip()

def _normalize_country(loc: str) -> str:
    us_states = {
        'CA','TX','NY','MA','VA','WA','FL','IL','CO','GA','NC','OH','PA',
        'MD','MN','MI','OR','AZ','NJ','UT','TN','IN','WI','MO','CT','NV',
        'KY','IA','OK','LA','SC','AL','AR','NE','MS','KS','ID','NH','ME',
        'HI','RI','DE','MT','SD','ND','AK','WY','VT','NM','WV'
    }
    parts = [p.strip() for p in loc.split(',')]
    last  = parts[-1].strip()
    if last in us_states or last in ('USA', 'United States', 'United States - Remote'):
        return 'United States'
    if last == 'California':                return 'United States'
    if 'Remote' in last and 'India' in loc: return 'India'
    if last == 'United Kingdom - Remote':   return 'United Kingdom'
    if last == 'Germany - Remote':          return 'Germany'
    if last == 'Brazil - Remote':           return 'Brazil'
    if last == 'Canada - Remote':           return 'Canada'
    if last == 'UK':                        return 'United Kingdom'
    return last

_all_countries = sorted(set(
    _normalize_country(loc)
    for loc in _df['Location'].dropna().unique()
    if _normalize_country(loc).strip()
))

# ── LangChain / Ollama setup ─────────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})

model = OllamaLLM(model="llama3.2", num_ctx=8192)

# ── Job matching prompt ───────────────────────────────────────────────────────
MATCH_PROMPT = ChatPromptTemplate.from_template("""
You are an expert career advisor specialising in data science and tech roles.

## Candidate Resume
{resume}

## Top Candidate Job Listings (retrieved by similarity)
{jobs}

## Instructions
Rank these jobs from best to worst fit for this candidate.
For EACH job output:

1. Rank # | Job Title | Company | Location
2. Match score: X/10
3. Matching skills (list 2-4 skills from the resume that align)
4. Skill gaps (list 1-3 requirements the candidate appears to lack)
5. One-sentence overall verdict.

Be specific — reference actual skills mentioned in the resume and job requirements.
""")

match_chain = MATCH_PROMPT | model

# ── Chatbot prompt ────────────────────────────────────────────────────────────
CHAT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert career coach and resume advisor specialising in data science and tech roles.

The candidate has shared their resume with you. Use it as context to answer their question.
Be specific, practical, and encouraging. Reference actual content from their resume when relevant.

## Candidate Resume
{resume}

## Conversation History
{history}

## Candidate's Question
{question}

## Your Answer
""")

chat_chain = CHAT_PROMPT | model

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_resume(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    text = text.strip()
    if not text:
        raise ValueError("Could not extract text from the uploaded file.")
    if len(text) > MAX_RESUME_CHARS:
        text = text[:MAX_RESUME_CHARS] + "\n\n[Resume truncated to fit context window]"
    return text


def location_matches(doc_location: str, selected: List[str]) -> bool:
    if not selected:
        return True
    return _normalize_country(doc_location) in selected


def format_jobs_for_llm(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        parts.append(
            f"[Job {i}]\n"
            f"Title:            {m.get('job_title', 'N/A')}\n"
            f"Company:          {m.get('company', 'N/A')}\n"
            f"Location:         {m.get('location', 'N/A')}\n"
            f"Job Type:         {m.get('job_type', 'N/A')}\n"
            f"Experience Level: {m.get('experience_level', 'N/A')}\n"
            f"Salary:           {m.get('salary', 'N/A')}\n"
            f"Required Skills:  {m.get('requirements', 'N/A')}\n"
            f"Facilities:       {m.get('facilities', 'N/A')}"
        )
    return "\n\n---\n\n".join(parts)


def format_history(history: list) -> str:
    """Format chat history list into a readable string for the prompt."""
    if not history:
        return "No previous messages."
    lines = []
    for msg in history:
        role = "Candidate" if msg["role"] == "user" else "Advisor"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/locations")
async def get_locations():
    return JSONResponse(content={"locations": _all_countries})


@app.post("/match")
async def match_resume(
    file: UploadFile = File(...),
    locations: str   = Form("[]"),
    session_id: str  = Form("default"),
):
    if not file.filename.lower().endswith((".pdf", ".txt")):
        return JSONResponse(status_code=400, content={"error": "Only .pdf and .txt files are supported."})

    selected_locations: List[str] = json.loads(locations)
    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        resume_text = load_resume(str(save_path))

        # Store resume in memory so the chatbot can reference it
        _resume_store[session_id] = resume_text

        resume_for_retrieval = resume_text[:2000]
        matched_docs = retriever.invoke(resume_for_retrieval)

        if not matched_docs:
            return JSONResponse(status_code=500, content={"error": "No jobs found. Run ingest_jobs.py first."})

        if selected_locations:
            filtered_docs = [
                doc for doc in matched_docs
                if location_matches(doc.metadata.get("location", ""), selected_locations)
            ]
            if not filtered_docs:
                filtered_docs = matched_docs
                location_note = "⚠️ No jobs found in selected locations — showing global results instead."
            else:
                location_note = None
        else:
            filtered_docs = matched_docs
            location_note = None

        show_top  = min(SHOW_TOP, len(filtered_docs))
        top_docs  = filtered_docs[:show_top]
        jobs_text = format_jobs_for_llm(top_docs)

        result = match_chain.invoke({"resume": resume_text, "jobs": jobs_text})

        job_cards = [
            {
                "title":    doc.metadata.get("job_title", "N/A"),
                "company":  doc.metadata.get("company",   "N/A"),
                "location": doc.metadata.get("location",  "N/A"),
                "salary":   doc.metadata.get("salary",    "N/A"),
                "exp":      doc.metadata.get("experience_level", "N/A"),
                "skills":   doc.metadata.get("requirements", "N/A"),
            }
            for doc in top_docs
        ]

        return JSONResponse(content={
            "analysis":      result,
            "jobs":          job_cards,
            "location_note": location_note,
            "session_id":    session_id,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        if save_path.exists():
            os.remove(save_path)


@app.post("/chat")
async def chat(
    session_id: str = Form(...),
    question:   str = Form(...),
    history:    str = Form("[]"),   # JSON array of {role, content} objects
):
    resume_text = _resume_store.get(session_id)

    if not resume_text:
        return JSONResponse(
            status_code=400,
            content={"error": "No resume found. Please upload your resume first using the matcher above."}
        )

    if not question.strip():
        return JSONResponse(status_code=400, content={"error": "Please enter a question."})

    try:
        history_list = json.loads(history)
        history_text = format_history(history_list)

        answer = chat_chain.invoke({
            "resume":   resume_text,
            "history":  history_text,
            "question": question.strip(),
        })

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})