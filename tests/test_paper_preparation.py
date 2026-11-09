import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.arxiv import ArxivError
from app.services.pdf import ExtractedPDFPage
from app.services.paper_preparation import (
    PaperPreparationError,
    _split_page_into_chunks,
    prepare_paper,
)


class PaperPreparationTests(unittest.TestCase):
    def test_long_page_splits_on_words_with_overlap_and_page_number(self):
        with (
            patch("app.services.paper_preparation.MAX_CHUNK_CHARACTERS", 16),
            patch("app.services.paper_preparation.CHUNK_OVERLAP_CHARACTERS", 6),
        ):
            chunks = _split_page_into_chunks(
                "alpha beta gamma delta epsilon", page_number=3, first_chunk_index=4
            )

        self.assertEqual(
            [(chunk.chunk_index, chunk.page_number, chunk.content) for chunk in chunks],
            [
                (4, 3, "alpha beta gamma"),
                (5, 3, "gamma delta"),
                (6, 3, "delta epsilon"),
            ],
        )

    @patch("app.services.paper_preparation._load_paper_pages")
    def test_preparation_numbers_chunks_across_pages(self, load_pages):
        load_pages.return_value = [
            ExtractedPDFPage(page_number=1, text="First page content"),
            ExtractedPDFPage(page_number=4, text="Fourth page content"),
        ]
        paper = {"source": "upload", "pdf_path": "stored.pdf"}

        chunks = prepare_paper(paper, Path("unused"))

        self.assertEqual(
            [(chunk.chunk_index, chunk.page_number, chunk.content) for chunk in chunks],
            [(0, 1, "First page content"), (1, 4, "Fourth page content")],
        )

    def test_missing_uploaded_pdf_reports_preparation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            paper = {"source": "upload", "pdf_path": "missing.pdf"}

            with self.assertRaisesRegex(PaperPreparationError, "could not be found"):
                prepare_paper(paper, directory)

    @patch("app.services.paper_preparation.download_paper_pdf")
    def test_arxiv_download_failure_reports_preparation_error(self, download):
        download.side_effect = ArxivError("offline")
        paper = {"source": "arxiv", "external_id": "2505.12345"}

        with self.assertRaisesRegex(PaperPreparationError, "could not be downloaded"):
            prepare_paper(paper, Path("unused"))
        download.assert_called_once_with("2505.12345")


if __name__ == "__main__":
    unittest.main()
