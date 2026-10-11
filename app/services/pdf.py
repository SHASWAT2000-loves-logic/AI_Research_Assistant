import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import fitz


class PDFProcessingError(Exception):
    """Raised when an uploaded PDF cannot be processed."""


@dataclass(frozen=True)
class UploadedPaperMetadata:
    stored_filename: str
    title: str
    authors: str
    publication_year: int | None
    abstract: str


@dataclass(frozen=True)
class ExtractedPDFPage:
    page_number: int
    text: str


def _clean_text(value):
    return " ".join((value or "").split())


def _normalize_latex_symbols(value):
    symbols = {
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "lambda": "λ",
        "mu": "μ",
        "pi": "π",
        "sigma": "σ",
        "phi": "φ",
        "psi": "ψ",
        "omega": "ω",
        "Omega": "Ω",
    }
    normalized = value or ""
    for command, symbol in symbols.items():
        normalized = re.sub(rf"\\{command}\b", symbol, normalized)
    return _clean_text(normalized.replace("$", "").replace("{", "").replace("}", ""))


def _title_tokens(value):
    ignored = {"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_latex_symbols(value).lower())
        if token not in ignored
    }


def _page_lines(page):
    lines = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = _clean_text("".join(span.get("text", "") for span in spans))
            if text:
                lines.append(
                    {
                        "text": text,
                        "size": max((span.get("size", 0) for span in spans), default=0),
                        "y": line.get("bbox", [0, 0, 0, 0])[1],
                    }
                )
    return sorted(lines, key=lambda item: item["y"])


def _layout_metadata(first_page, embedded_title=""):
    lines = _page_lines(first_page)
    if not lines:
        return "", ""

    abstract_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line["text"].strip().lower() == "abstract"
        ),
        None,
    )
    lines_before_abstract = lines[:abstract_index] if abstract_index is not None else lines
    top_lines = [
        line
        for line in lines_before_abstract
        if line["y"] < first_page.rect.height * 0.5
        and not line["text"].lower().startswith("arxiv:")
    ]
    if not top_lines:
        top_lines = lines[:10]

    embedded_tokens = _title_tokens(embedded_title)
    title_lines = []
    if embedded_tokens:
        for line in top_lines:
            line_tokens = _title_tokens(line["text"])
            if not line_tokens:
                continue
            overlap = len(line_tokens & embedded_tokens) / min(
                len(line_tokens), len(embedded_tokens)
            )
            if overlap >= 0.5:
                title_lines.append(line)

    if not title_lines:
        largest_size = max(line["size"] for line in top_lines)
        title_lines = [
            line
            for line in top_lines
            if line["size"] >= largest_size - 0.5 and len(line["text"]) > 3
        ]

    title = _clean_text(" ".join(line["text"] for line in title_lines))

    title_bottom = max((line["y"] for line in title_lines), default=-1)
    author_candidates = []
    excluded_words = (
        "abstract",
        "university",
        "department",
        "institute",
        "laboratory",
        "@",
        "http",
        "keywords",
    )
    for line in top_lines:
        lowered = line["text"].lower()
        if line["y"] <= title_bottom:
            continue
        if "abstract" in lowered:
            break
        if any(word in lowered for word in excluded_words):
            if author_candidates:
                break
            continue
        if re.fullmatch(r"[\d\W_]+", line["text"]):
            continue
        author_candidates.append(line["text"])
        if len(author_candidates) == 2:
            break

    return title, _clean_text(", ".join(author_candidates))


def _extract_year(document_metadata, first_page_text):
    arxiv_line = re.search(r"(?im)^\s*arxiv:[^\n]+$", first_page_text)
    if arxiv_line:
        explicit_year = re.search(r"\b((?:19|20)\d{2})\b", arxiv_line.group(0))
        if explicit_year:
            return int(explicit_year.group(1))

        modern_identifier = re.search(r"arxiv:(\d{2})(\d{2})\.\d+", arxiv_line.group(0), re.I)
        if modern_identifier:
            month = int(modern_identifier.group(2))
            if 1 <= month <= 12:
                return 2000 + int(modern_identifier.group(1))

    date_patterns = (
        r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+((?:19|20)\d{2})\b",
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+((?:19|20)\d{2})\b",
        r"\b(?:published|publication|copyright|proceedings)\b[^\n]{0,80}\b((?:19|20)\d{2})\b",
    )
    for pattern in date_patterns:
        match = re.search(pattern, first_page_text, re.I)
        if match:
            return int(match.group(1))

    for field in ("creationDate", "modDate"):
        match = re.search(r"((?:19|20)\d{2})", document_metadata.get(field, ""))
        if match:
            return int(match.group(1))

    return None


def _clean_wrapped_text(value):
    hyphenated_left_parts = {
        "anti",
        "context",
        "cross",
        "data",
        "decision",
        "domain",
        "high",
        "human",
        "ill",
        "left",
        "long",
        "low",
        "machine",
        "model",
        "multi",
        "non",
        "policy",
        "post",
        "pre",
        "quasi",
        "real",
        "right",
        "self",
        "semi",
        "short",
        "state",
        "sub",
        "super",
        "task",
        "time",
        "trans",
        "user",
        "well",
    }

    def join_word(match):
        left, right = match.group(1), match.group(2)
        preserve_hyphen = (
            left.lower() in hyphenated_left_parts
            or left.isupper()
            or any(ord(character) > 127 for character in left)
        )
        separator = "-" if preserve_hyphen else ""
        return f"{left}{separator}{right}"

    cleaned = re.sub(
        r"([^\W\d_]+)-[ \t]*\n[ \t]*([^\W\d_]+)",
        join_word,
        value,
    )
    return _clean_text(cleaned)


def _extract_abstract(text):
    match = re.search(
        r"(?is)\babstract\s*[.:\-\u2014]?\s*(.+?)"
        r"(?=\n\s*(?:keywords?\b|(?:(?:1|i)[\.\s]+)?introduction\b))",
        text,
    )
    if not match:
        return ""
    return _clean_wrapped_text(match.group(1))


def _document_pages(document):
    pages = []
    for page_index, page in enumerate(document):
        text = _clean_wrapped_text(page.get_text("text"))
        if text:
            pages.append(
                ExtractedPDFPage(
                    page_number=page_index + 1,
                    text=text,
                )
            )

    if not pages:
        raise PDFProcessingError(
            "No selectable text was found in this paper, so it cannot be prepared."
        )
    return pages


def extract_pdf_pages(path):
    try:
        document = fitz.open(path)
    except (fitz.FileDataError, RuntimeError) as error:
        raise PDFProcessingError("The saved PDF could not be read.") from error

    try:
        return _document_pages(document)
    finally:
        document.close()


def extract_pdf_pages_bytes(pdf_data):
    try:
        document = fitz.open(stream=pdf_data, filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as error:
        raise PDFProcessingError("The downloaded PDF could not be read.") from error

    try:
        return _document_pages(document)
    finally:
        document.close()


def _extract_metadata(path, stored_filename):
    try:
        document = fitz.open(path)
    except (fitz.FileDataError, RuntimeError) as error:
        raise PDFProcessingError("The selected file is not a readable PDF.") from error

    try:
        if document.page_count == 0:
            raise PDFProcessingError("The PDF does not contain any pages.")

        page_texts = [page.get_text("text") for page in document]
        full_text = "\n".join(page_texts)
        if not full_text.strip():
            raise PDFProcessingError(
                "No selectable text was found. Please upload a text-based PDF."
            )

        metadata = document.metadata or {}
        embedded_title = _normalize_latex_symbols(metadata.get("title"))
        layout_title, layout_authors = _layout_metadata(document[0], embedded_title)
        title = layout_title or embedded_title
        authors = _clean_text(metadata.get("author")) or layout_authors

        return UploadedPaperMetadata(
            stored_filename=stored_filename,
            title=title,
            authors=authors,
            publication_year=_extract_year(metadata, page_texts[0]),
            abstract=_extract_abstract(full_text),
        )
    finally:
        document.close()


def process_uploaded_pdf(uploaded_file, upload_folder):
    if not uploaded_file.filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Choose a file with a .pdf extension.")

    header = uploaded_file.stream.read(5)
    uploaded_file.stream.seek(0)
    if header != b"%PDF-":
        raise PDFProcessingError("The selected file is not a valid PDF.")

    stored_filename = f"{uuid4().hex}.pdf"
    destination = Path(upload_folder) / stored_filename
    uploaded_file.save(destination)

    try:
        return _extract_metadata(destination, stored_filename)
    except PDFProcessingError:
        destination.unlink(missing_ok=True)
        raise
