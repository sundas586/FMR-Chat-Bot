# FMR RAG Chatbot Learning Project

A small, teachable Retrieval-Augmented Generation prototype over the two supplied NBP Fund Manager Reports (FMRs) for July 2026:

- `Complete-FMR-Conventional-July-2026.pdf`
- `Complete-FMR-Islamic-July-2026.pdf`

The project extracts PDF text page by page, stores chunks and metadata in ChromaDB, retrieves relevant evidence, and optionally asks an OpenAI model to write a grounded answer with citations.

## Architecture

```text
PDFs -> PyMuPDF extraction -> page-aware chunks + metadata -> ChromaDB
                                                           |
User question -> similarity retrieval -> grounded prompt -> OpenAI answer
                                          |
                                    Streamlit chat UI
```

Each chunk records `category`, `report_month`, `report_year`, `source_file`, `source_page`, and `section`. This keeps Conventional and Islamic fund content distinguishable and makes future PDFs a drop-in ingestion operation.

## Setup

From the project folder:

```powershell
cd d:\Copilot-MCP\projects\Learning_Project
..\..\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

The default client-facing model is local Ollama, so no OpenAI key or credits are required. Install Ollama, download the configured model, and keep the Ollama service running. OpenAI remains an optional provider if `LLM_PROVIDER=openai` is selected.

Install Ollama and download the model:

```powershell
winget install Ollama.Ollama
ollama pull qwen2.5:7b
```

## Run

Build the local vector database after adding or replacing PDFs:

```powershell
..\..\.venv\Scripts\python.exe ingest.py --reset
```

Start the API:

```powershell
uvicorn main:app --reload
```

Start the chat UI in a second terminal:

```powershell
streamlit run streamlit_app.py
```

Open the Streamlit URL shown in the terminal. The sidebar can re-ingest the PDFs after new reports are dropped into this folder.

## API examples

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Ask a question:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chat -ContentType 'application/json' -Body '{"question":"What was NBP Stock Fund''s return in July 2026?"}'
```

## Accuracy notes

- Answers are restricted to retrieved PDF context and use temperature `0`.
- The system refuses to guess when the context does not contain an answer.
- Source metadata is returned with every response for page-level verification.
- This is a learning prototype, not investment advice or a production financial system.
- For production use, add table-aware extraction, a fund-name metadata parser, reranking, evaluation questions, authentication, and an audit trail.
