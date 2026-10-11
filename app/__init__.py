import os
from pathlib import Path

from flask import (
    Flask,
    abort,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from . import db
from .papers import (
    get_paper,
    get_paper_chunks,
    get_paper_summary,
    list_papers,
    replace_paper_chunks,
    save_arxiv_paper,
    save_paper_summary,
    save_uploaded_paper,
)
from .services.pdf import PDFProcessingError, process_uploaded_pdf
from .services.arxiv import ArxivError, search_papers
from .services.llm import PaperAIError, answer_paper_question, generate_paper_summary
from .services.paper_preparation import PaperPreparationError, prepare_paper


def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        DATABASE=str(Path(app.instance_path) / "research_assistant.sqlite3"),
        UPLOAD_FOLDER=str(Path(app.instance_path) / "uploads"),
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        GEMINI_MODEL=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    )

    if test_config:
        app.config.update(test_config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    db.init_app(app)

    @app.get("/")
    def index():
        return redirect(url_for("search"))

    @app.get("/search")
    def search():
        query = request.args.get("query", "").strip()
        results = None
        error = None

        if "query" in request.args:
            if not query:
                error = "Enter a research topic or keyword to search."
            else:
                try:
                    results = search_papers(query)
                except ArxivError:
                    error = "arXiv search is temporarily unavailable. Please try again."

        return render_template(
            "search.html",
            query=query,
            results=results,
            error=error,
        )

    @app.get("/library")
    def library():
        status = request.args.get("status")
        message = {
            "saved": "Paper saved to your library.",
            "exists": "This paper is already in your library.",
        }.get(status)
        return render_template(
            "library.html",
            papers=list_papers(),
            message=message,
        )

    @app.post("/library/save")
    def save_to_library():
        external_id = request.form.get("external_id", "").strip()
        title = request.form.get("title", "").strip()
        if not external_id or not title:
            abort(400)

        year_value = request.form.get("publication_year", "").strip()
        try:
            publication_year = int(year_value) if year_value else None
        except ValueError:
            publication_year = None

        created = save_arxiv_paper(
            external_id=external_id,
            title=title,
            authors=request.form.get("authors", "").strip(),
            publication_year=publication_year,
            abstract=request.form.get("abstract", "").strip(),
            paper_url=request.form.get("paper_url", "").strip(),
        )
        status = "saved" if created else "exists"
        return redirect(url_for("library", status=status))

    @app.get("/papers/<int:paper_id>")
    def paper_detail(paper_id):
        paper = get_paper(paper_id)
        if paper is None:
            abort(404)
        return render_template(
            "paper_detail.html",
            paper=paper,
            preparation_message=(
                f"Paper prepared successfully: {request.args.get('prepared', type=int)} "
                "text chunks saved."
                if request.args.get("prepared", type=int) is not None
                else None
            ),
            preparation_error=None,
            summary=get_paper_summary(paper_id),
            summary_message=(
                "AI summary generated successfully."
                if request.args.get("summarized") == "1"
                else None
            ),
            summary_error=None,
        )

    @app.post("/papers/<int:paper_id>/prepare")
    def prepare_saved_paper(paper_id):
        paper = get_paper(paper_id)
        if paper is None:
            abort(404)

        try:
            chunks = prepare_paper(
                paper,
                Path(app.config["UPLOAD_FOLDER"]),
            )
            chunk_count = replace_paper_chunks(paper_id, chunks)
        except PaperPreparationError as error:
            return render_template(
                "paper_detail.html",
                paper=paper,
                preparation_message=None,
                preparation_error=str(error),
                summary=get_paper_summary(paper_id),
                summary_message=None,
                summary_error=None,
            )

        return redirect(
            url_for("paper_detail", paper_id=paper_id, prepared=chunk_count)
            + "#preparation"
        )

    @app.post("/papers/<int:paper_id>/summary")
    def summarize_saved_paper(paper_id):
        paper = get_paper(paper_id)
        if paper is None:
            abort(404)

        chunks = get_paper_chunks(paper_id)
        if not chunks:
            return render_template(
                "paper_detail.html",
                paper=paper,
                preparation_message=None,
                preparation_error=None,
                summary=get_paper_summary(paper_id),
                summary_message=None,
                summary_error="Prepare this paper before generating a summary.",
            )

        try:
            summary_text = generate_paper_summary(
                title=paper["title"],
                authors=paper["authors"],
                chunks=chunks,
                model=app.config["GEMINI_MODEL"],
            )
        except PaperAIError as error:
            return render_template(
                "paper_detail.html",
                paper=paper,
                preparation_message=None,
                preparation_error=None,
                summary=get_paper_summary(paper_id),
                summary_message=None,
                summary_error=str(error),
            )

        save_paper_summary(
            paper_id=paper_id,
            summary=summary_text,
            model=app.config["GEMINI_MODEL"],
        )
        return redirect(
            url_for("paper_detail", paper_id=paper_id, summarized="1")
            + "#summary"
        )

    @app.post("/papers/<int:paper_id>/ask")
    def ask_saved_paper(paper_id):
        paper = get_paper(paper_id)
        if paper is None:
            abort(404)

        question = request.form.get("question", "").strip()
        question_error = None
        if not question:
            question_error = "Enter a question about this paper."
        elif len(question) > 1_000:
            question_error = "Keep the question under 1,000 characters."

        chunks = get_paper_chunks(paper_id)
        if not chunks:
            question_error = "Prepare this paper before asking questions about it."

        answer = None
        if question_error is None:
            try:
                answer = answer_paper_question(
                    title=paper["title"],
                    question=question,
                    chunks=chunks,
                    model=app.config["GEMINI_MODEL"],
                )
            except PaperAIError as error:
                question_error = str(error)

        return render_template(
            "paper_detail.html",
            paper=paper,
            preparation_message=None,
            preparation_error=None,
            summary=get_paper_summary(paper_id),
            summary_message=None,
            summary_error=None,
            question=question,
            answer=answer,
            question_error=question_error,
        )

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html")

        uploaded_file = request.files.get("paper")
        if uploaded_file is None or not uploaded_file.filename:
            return render_template(
                "upload.html",
                error="Choose a PDF file to upload.",
            ), 400

        try:
            metadata = process_uploaded_pdf(
                uploaded_file,
                Path(app.config["UPLOAD_FOLDER"]),
            )
        except PDFProcessingError as error:
            return render_template("upload.html", error=str(error)), 400

        return render_template("upload_confirm.html", metadata=metadata)

    @app.post("/upload/confirm")
    def confirm_upload():
        stored_filename = request.form.get("stored_filename", "").strip()
        title = request.form.get("title", "").strip()
        if not title or not _is_valid_stored_pdf(stored_filename):
            abort(400)

        upload_path = Path(app.config["UPLOAD_FOLDER"]) / stored_filename
        if not upload_path.is_file():
            abort(400)

        year_value = request.form.get("publication_year", "").strip()
        try:
            publication_year = int(year_value) if year_value else None
        except ValueError:
            publication_year = None

        paper_id = save_uploaded_paper(
            title=title,
            authors=request.form.get("authors", "").strip(),
            publication_year=publication_year,
            abstract=request.form.get("abstract", "").strip(),
            stored_filename=stored_filename,
        )
        return redirect(url_for("paper_detail", paper_id=paper_id))

    @app.get("/papers/<int:paper_id>/pdf")
    def paper_pdf(paper_id):
        paper = get_paper(paper_id)
        if paper is None or not paper["pdf_path"]:
            abort(404)
        if not _is_valid_stored_pdf(paper["pdf_path"]):
            abort(404)
        return send_from_directory(
            app.config["UPLOAD_FOLDER"],
            paper["pdf_path"],
            mimetype="application/pdf",
        )

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_error):
        return render_template(
            "upload.html",
            error="The PDF is too large. Choose a file smaller than 20 MB.",
        ), 413

    return app


def _is_valid_stored_pdf(filename):
    return (
        bool(filename)
        and Path(filename).name == filename
        and filename.lower().endswith(".pdf")
    )
