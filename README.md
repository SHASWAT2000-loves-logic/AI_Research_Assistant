# AI Research Assistant

This Flask application lets you search arXiv, save papers in a local SQLite library, upload text-based PDFs, summarize prepared papers with Gemini, and ask questions about their contents.

## Features

- Search arXiv by keyword or topic; view titles, authors, publication years, abstracts, and paper links.
- Save search results and browse them after a refresh or restart.
- Upload a PDF, review and edit its extracted title, authors, year, and abstract, then save it.
- Prepare a saved paper by extracting page-numbered PDF text into SQLite chunks. Saved arXiv PDFs are downloaded when you prepare them; uploads use their stored PDF.
- Generate and save an AI summary from prepared text, or ask a question and receive an answer with requested page citations.

arXiv requests are spaced at least three seconds apart. Each summary or question makes one Gemini API request.

## Setup

Use Python 3.10 or newer. In Windows PowerShell, run these commands from the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GEMINI_API_KEY="your-api-key"
.\.venv\Scripts\python.exe run.py
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"
.venv/bin/python run.py
```

Open <http://127.0.0.1:5000>. Set `GEMINI_API_KEY` in the same terminal used to start the app. The key is needed for summaries and questions; search, saving, upload, and preparation work without it. The default model is `gemini-3.5-flash-lite`; set `GEMINI_MODEL` before starting the app if you use another available Gemini model. Internet access is needed for arXiv search, arXiv PDF preparation, and Gemini requests. Do not put API keys in source files or commits.

The app creates `instance/research_assistant.sqlite3` and stores uploaded PDFs in `instance/uploads/`. Both are local data and are ignored by Git. PDF uploads must contain selectable text; scanned PDFs are not supported. Extracted metadata is best effort and can be corrected before saving.

## Project structure

- `run.py`: starts the Flask app.
- `app/__init__.py`: routes and application setup.
- `app/db.py`, `app/schema.sql`, `app/papers.py`: SQLite setup and paper storage.
- `app/services/arxiv.py`: arXiv search and PDF download.
- `app/services/pdf.py`, `app/services/paper_preparation.py`: PDF processing and page-numbered chunks.
- `app/services/llm.py`: Gemini summary and paper question prompts.
- `app/templates/`, `app/static/`: user interface.

## Check and demonstrate

Run automated checks with `.\.venv\Scripts\python.exe -m unittest discover -s tests -v` on Windows, or `.venv/bin/python -m unittest discover -s tests -v` on macOS/Linux. The workflow test uses a temporary database and mocks external services, so it does not change your Library or spend API quota.

For the required video, demonstrate searching, saving, uploading, summarizing, and asking a paper question. Compare AI summaries and answers with the cited PDF pages; model output is not guaranteed to be correct.
