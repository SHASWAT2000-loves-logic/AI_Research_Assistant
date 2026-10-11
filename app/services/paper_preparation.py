from dataclasses import dataclass
from pathlib import Path

from .arxiv import ArxivError, download_paper_pdf
from .pdf import PDFProcessingError, extract_pdf_pages, extract_pdf_pages_bytes


MAX_CHUNK_CHARACTERS = 3_000
CHUNK_OVERLAP_CHARACTERS = 300


class PaperPreparationError(Exception):
    """Raised when a saved paper cannot be prepared for AI features."""


@dataclass(frozen=True)
class PreparedPaperChunk:
    chunk_index: int
    page_number: int
    content: str


def _split_page_into_chunks(page_text, page_number, first_chunk_index):
    words = page_text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start
        character_count = 0
        while end < len(words):
            additional_length = len(words[end]) + (1 if end > start else 0)
            if character_count + additional_length > MAX_CHUNK_CHARACTERS:
                break
            character_count += additional_length
            end += 1

        if end == start:
            end += 1

        chunks.append(
            PreparedPaperChunk(
                chunk_index=first_chunk_index + len(chunks),
                page_number=page_number,
                content=" ".join(words[start:end]),
            )
        )

        if end == len(words):
            break

        overlap_start = end
        overlap_length = 0
        while overlap_start > start:
            word_length = len(words[overlap_start - 1]) + 1
            if overlap_length + word_length > CHUNK_OVERLAP_CHARACTERS:
                break
            overlap_start -= 1
            overlap_length += word_length
        start = overlap_start if overlap_start > start else end

    return chunks


def _load_paper_pages(paper, upload_folder):
    if paper["source"] == "upload" and paper["pdf_path"]:
        stored_filename = paper["pdf_path"]
        if Path(stored_filename).name != stored_filename:
            raise PaperPreparationError("The saved PDF path is invalid.")

        pdf_path = Path(upload_folder) / stored_filename
        if not pdf_path.is_file():
            raise PaperPreparationError("The uploaded PDF could not be found.")

        try:
            return extract_pdf_pages(pdf_path)
        except PDFProcessingError as error:
            raise PaperPreparationError(str(error)) from error

    if paper["source"] == "arxiv" and paper["external_id"]:
        try:
            return extract_pdf_pages_bytes(download_paper_pdf(paper["external_id"]))
        except ArxivError as error:
            raise PaperPreparationError(
                "The paper could not be downloaded from arXiv. Please try again."
            ) from error
        except PDFProcessingError as error:
            raise PaperPreparationError(str(error)) from error

    raise PaperPreparationError("No PDF is available for this library item.")


def prepare_paper(paper, upload_folder):
    pages = _load_paper_pages(paper, upload_folder)
    chunks = []
    for page in pages:
        page_chunks = _split_page_into_chunks(
            page.text,
            page.page_number,
            len(chunks),
        )
        chunks.extend(page_chunks)

    if not chunks:
        raise PaperPreparationError("No text chunks could be created from this paper.")
    return chunks
