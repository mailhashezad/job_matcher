# 🔍 Data Science Job Matcher

An AI-powered local job matching system that takes your resume and finds the most relevant data science roles from a dataset of 3,000+ job listings — with a built-in resume chatbot for career coaching. Everything runs **100% locally** using Ollama, no API keys or internet connection required.

---

## 📸 Screenshots

### Job Matcher


![Job Matcher UI](screenshots/job_matcher.png)

### Location Filter


![Location Filter](screenshots/location_filter.mp4)

### Job Results


![Job Results](screenshots/job_results.mp4)

### Resume Chatbot


![Resume Chatbot](screenshots/chatbot.mp4)

---

## 🧠 How It Works

The system is built on a **Retrieval-Augmented Generation (RAG)** pipeline:

![system architecture](screenshots/system_architecture.svg)

The **chatbot** uses the same llama3.2 model. Once a resume is uploaded, it's stored in memory for the session and used as context for every chat message, enabling personalised multi-turn career coaching.




---

## 🗂️ Project Structure

```
job_matcher/
├── app.py                  # FastAPI backend (job matching + chatbot API)
├── index.html              # Frontend UI (single file, no framework)
├── ingest_jobs.py          # One-time script to index jobs into Chroma
├── resume_matcher.py       # CLI version of the matcher
├── data_science_job.csv    # Dataset of 3,198 data science job listings
├── jobs_db/                # Chroma vector store (auto-created after ingestion)
├── uploads/                # Temporary resume storage (auto-cleaned)
├── .gitignore
└── README.md
```

---

## 📦 Dataset

The job listings dataset (`data_science_job.csv`) contains **3,198 data science roles** with the following columns:

| Column | Description |
|---|---|
| `Company` | Hiring company name |
| `Job Title` | Role title |
| `Location` | City, country |
| `Job Type` | Full-time, contract, etc. |
| `Experience level` | Entry, mid, senior |
| `Salary` | Salary range |
| `Requirment of the company` | Comma-separated required skills (used as job description) |
| `Facilities` | Benefits and perks |

The `Requirment of the company` column is used as the **job description** for embedding and retrieval.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | llama3.2 via Ollama |
| Embeddings | mxbai-embed-large via Ollama |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |
| PDF Parsing | pypdf |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com) installed and running

### Step 1 — Clone the repo
```bash
git clone https://github.com/yourusername/job-matcher.git
cd job-matcher
```

### Step 2 — Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install langchain langchain-ollama langchain-chroma chromadb pypdf pandas fastapi uvicorn python-multipart
```

### Step 4 — Pull Ollama models
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Step 5 — Index the job listings (run once)
```bash
python ingest_jobs.py
```
This embeds all 3,198 jobs into a local Chroma vector store. It takes **5–15 minutes** depending on your machine. Once complete, a `jobs_db/` folder is created — you never need to run this again.

### Step 6 — Start the server
```bash
uvicorn app:app --reload
```

### Step 7 — Open in browser
```
http://localhost:8000
```

---

## 🖥️ Usage

### Job Matcher Tab
1. Drag and drop or click to upload your resume (`.pdf` or `.txt`)
2. Optionally select one or more preferred countries from the location dropdown
3. Click **Find Matching Jobs**
4. View the top matched job cards and the AI's ranked analysis with match scores and skill gaps

### Resume Chatbot Tab
1. Upload your resume via the Job Matcher tab first
2. Switch to the **Resume Chatbot** tab
3. Use the suggestion chips or type your own question
4. The AI responds based on your actual resume content

Example questions you can ask:
- *"How can I improve my resume?"*
- *"What skills am I missing for ML engineer roles?"*
- *"Write a professional summary for me"*
- *"Give me interview tips based on my background"*

### CLI Usage (optional)
```bash
python resume_matcher.py your_resume.pdf
```

---

## ⚠️ Limitations

### 1. Context Window Constraint
`mxbai-embed-large` has a 512-token embedding limit. Resumes are truncated to the first **2,000 characters** for the retrieval step. This means only the top portion of the resume is used to find matching jobs — skills buried further down may not influence retrieval.

### 2. No Persistent Chat Sessions
Resume data is stored **in memory only**. If you restart the server, the resume is lost and must be re-uploaded. There is no database or file-based session persistence.

### 3. Local Hardware Dependency
Performance depends entirely on your machine. On CPU-only machines, each query can take **30–90 seconds**. Users with a dedicated GPU will see significantly faster results.

### 4. Dataset Quality
The job requirements are stored as comma-separated skill keywords (e.g., `"Python, AWS, SQL"`), not full natural language descriptions. This limits the semantic richness of the embeddings and can affect match accuracy.

### 5. Single-User Design
The session system uses a simple timestamp ID with no authentication. It is **not suitable for multi-user deployment** — multiple users on the same server instance would not have isolated sessions.

### 6. Location Filtering is Post-Retrieval
Locations are filtered **after** vector retrieval, not during. If you select a rare location, the system may fall back to global results because the retrieved pool didn't contain enough local matches.

### 7. No Resume Parsing Structure
Resumes are treated as raw plain text. The system doesn't parse sections like Education, Experience, or Skills separately — it treats the entire resume as one blob of text.

---

## 🔮 Future Improvements

### Short Term
- [ ] **Structured resume parsing** — extract sections (Skills, Experience, Education) separately and weight them differently during retrieval
- [ ] **GPU acceleration** — detect and use CUDA/Metal for faster Ollama inference
- [ ] **Resume upload in chatbot tab** — allow uploading directly from the chat tab without switching
- [ ] **Export results** — download matched jobs as a PDF or CSV

### Medium Term
- [ ] **Persistent sessions** — store resume data in SQLite or Redis so sessions survive server restarts
- [ ] **Live job listings** — integrate a jobs API (Adzuna, RapidAPI Jobs, or LinkedIn scraper) to replace the static CSV with real-time data
- [ ] **Better embedding model** — replace `mxbai-embed-large` with `nomic-embed-text` or a model with a larger context window to avoid truncation
- [ ] **Job bookmarking** — let users save and compare favourite job matches
- [ ] **Match history** — store past searches so users can track how their resume performs over time

### Long Term
- [ ] **Multi-user support** — add authentication (FastAPI + JWT) and per-user session isolation
- [ ] **Resume builder** — let the AI generate an improved version of the resume based on chatbot feedback
- [ ] **Interview prep mode** — generate likely interview questions based on the job description and the user's resume
- [ ] **Salary insights** — analyse salary ranges across matched jobs and compare to the user's experience level
- [ ] **Cloud deployment** — containerise with Docker and deploy to a cloud platform (Railway, Render, or AWS)

---

## 📁 .gitignore

The following are excluded from version control:

```
venv/          # virtual environment
jobs_db/       # Chroma vector store (regenerated by ingest_jobs.py)
uploads/       # temporary resume files
__pycache__/   # Python bytecode
*.pyc
.env           # secrets and environment variables
```

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com) — local LLM runtime
- [LangChain](https://langchain.com) — LLM orchestration framework
- [ChromaDB](https://trychroma.com) — local vector store
- [FastAPI](https://fastapi.tiangolo.com) — Python web framework

---

## 📄 Conclusion

This project demonstrates how to build a **fully local, privacy-preserving AI application** without relying on any paid APIs or cloud services. By combining vector similarity search with an LLM ranking step, the system goes beyond simple keyword matching to understand the *semantic* fit between a candidate's background and job requirements.

The resume chatbot adds a practical layer on top — rather than just showing results, it lets candidates have a real conversation about their career, get actionable feedback on their resume, and understand exactly where their skill gaps lie.

While the current version is designed as a single-user local tool, the architecture is solid enough to scale — swapping the static CSV for a live jobs API, adding authentication, and deploying to a cloud platform would turn it into a production-ready career tool.

---

> Built with ❤️ using LangChain, Ollama, ChromaDB, and FastAPI.
