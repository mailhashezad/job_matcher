"""
ingest_jobs.py — Run once to index the data science job listings into Chroma.
Reads from data_science_job.csv and uses the 'Requirment of the company'
column as the job description for embedding and retrieval.
"""

import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH    = "data_science_job.csv"
DB_LOCATION = "./jobs_db"
COLLECTION  = "data_science_jobs"

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="latin-1")

# Normalise column names (CSV has trailing spaces)
df.columns = df.columns.str.strip()

# Drop rows with no requirements or no job title
df = df.dropna(subset=["Requirment of the company", "Job Title"])
df = df.reset_index(drop=True)

print(f"Loaded {len(df)} job listings from '{CSV_PATH}'.")

# ── Build Documents ───────────────────────────────────────────────────────────
documents = []
ids = []

for i, row in df.iterrows():
    # Clean requirements: "AWS,Azure,Python,," → "AWS, Azure, Python"
    raw_req = str(row["Requirment of the company"]).strip()
    skills  = [s.strip() for s in raw_req.split(",") if s.strip()]
    requirements = ", ".join(skills)

    # page_content = what gets embedded (title + skills)
    page_content = (
        f"Job Title: {row['Job Title']}\n"
        f"Required Skills: {requirements}"
    )

    doc = Document(
        page_content=page_content,
        metadata={
            "job_title":        str(row.get("Job Title",        "N/A")),
            "company":          str(row.get("Company",          "N/A")),
            "location":         str(row.get("Location",         "N/A")),
            "job_type":         str(row.get("Job Type",         "N/A")),
            "experience_level": str(row.get("Experience level", "N/A")),
            "salary":           str(row.get("Salary",           "N/A")),
            "facilities":       str(row.get("Facilities",       "N/A")),
            "requirements":     requirements,
        },
        id=str(i),
    )
    documents.append(doc)
    ids.append(str(i))

# ── Index into Chroma ─────────────────────────────────────────────────────────
if os.path.exists(DB_LOCATION) and os.listdir(DB_LOCATION):
    print(f"Vector store already exists at '{DB_LOCATION}'. Delete the folder to re-index.")
else:
    print("Embedding and indexing jobs — this may take a few minutes on 3k rows...")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vector_store = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_LOCATION,
        embedding_function=embeddings,
    )

    # Batch to avoid memory spikes
    BATCH = 200
    for start in range(0, len(documents), BATCH):
        batch_docs = documents[start : start + BATCH]
        batch_ids  = ids[start : start + BATCH]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"  Indexed {min(start + BATCH, len(documents))} / {len(documents)} jobs...")

    print(f"\nDone. {len(documents)} jobs indexed into '{DB_LOCATION}'.")
