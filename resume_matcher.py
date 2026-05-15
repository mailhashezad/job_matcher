"""
resume_matcher.py — Match a candidate's resume against data science job listings.

Usage:
    python resume_matcher.py resume.pdf
    python resume_matcher.py resume.txt
"""

import argparse
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────
DB_LOCATION = "./jobs_db"
COLLECTION  = "data_science_jobs"
TOP_K       = 7    # jobs retrieved from vector store before LLM ranking
SHOW_TOP    = 5    # how many the LLM should rank and explain


# ── Resume loading ────────────────────────────────────────────────────────────

def load_resume(path: str) -> str:
    """Extract plain text from a .pdf or .txt resume."""
    if path.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("Run: pip install pypdf")
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    if not text.strip():
        raise ValueError(f"No text could be extracted from '{path}'.")
    return text.strip()


# ── Vector retrieval ──────────────────────────────────────────────────────────

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

vector_store = Chroma(
    collection_name=COLLECTION,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})


# ── LLM ranking ───────────────────────────────────────────────────────────────

model = OllamaLLM(model="llama3.2")

PROMPT_TEMPLATE = """
You are an expert career advisor specialising in data science and tech roles.

## Candidate Resume
{resume}

## Top {show_top} Candidate Job Listings (retrieved by similarity)
{jobs}

## Instructions
Rank these {show_top} jobs from best to worst fit for this candidate.
For EACH job output:

1. Rank # | Job Title | Company | Location
2. Match score: X/10
3. ✅ Matching skills (list 2–4 skills from the resume that align)
4. ⚠️  Skill gaps (list 1–3 requirements the candidate appears to lack)
5. One-sentence overall verdict.

Be specific — reference actual skills mentioned in the resume and job requirements.
If the candidate's resume is vague, say so.
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
chain  = prompt | model


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_jobs_for_llm(docs) -> str:
    """Convert retrieved Documents into a readable block for the LLM."""
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


def print_header(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ── Main ──────────────────────────────────────────────────────────────────────

def match_resume(resume_path: str):
    print_header("DATA SCIENCE JOB MATCHER")

    # 1. Load resume
    print(f"\n📄 Loading resume: {resume_path}")
    resume_text = load_resume(resume_path)
    preview = resume_text[:200].replace("\n", " ")
    print(f"   Preview: {preview}...")
    print(f"   Total length: {len(resume_text)} characters\n")

    # 2. Retrieve candidate jobs
    print(f"🔍 Searching {TOP_K} most relevant jobs from the database...")
    matched_docs = retriever.invoke(resume_text)

    if not matched_docs:
        print("❌ No jobs found. Run ingest_jobs.py first.")
        return

    # Show quick list of retrieved jobs
    print(f"\n   Retrieved candidates:")
    for i, doc in enumerate(matched_docs, 1):
        m = doc.metadata
        print(f"   {i}. {m.get('job_title','?')} @ {m.get('company','?')} — {m.get('location','?')}")

    # 3. LLM ranking
    show_top = min(SHOW_TOP, len(matched_docs))
    top_docs = matched_docs[:show_top]
    jobs_text = format_jobs_for_llm(top_docs)

    print(f"\n🤖 Asking LLM to rank top {show_top} jobs...\n")

    result = chain.invoke({
        "resume":    resume_text,
        "jobs":      jobs_text,
        "show_top":  show_top,
    })

    print_header("RANKED JOB MATCHES")
    print(result)
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Match a resume to data science job listings."
    )
    parser.add_argument("resume", help="Path to resume (.pdf or .txt)")
    args = parser.parse_args()
    match_resume(args.resume)
