import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import fitz
from werkzeug.datastructures import FileStorage

from app.services.pdf import (
    PDFProcessingError,
    _clean_wrapped_text,
    _extract_abstract,
    _extract_year,
    _normalize_latex_symbols,
    extract_pdf_pages_bytes,
    process_uploaded_pdf,
)


class PDFProcessingTests(unittest.TestCase):
    def test_wrapped_words_preserve_genuine_hyphens(self):
        text = "formal verifica-\n tion and non-\n deterministic ω-\n regular right-\n linear models"

        self.assertEqual(
            _clean_wrapped_text(text),
            "formal verification and non-deterministic ω-regular right-linear models",
        )

    def test_arxiv_year_overrides_stale_embedded_pdf_date(self):
        metadata = {"creationDate": "D:20120101000000"}

        self.assertEqual(
            _extract_year(metadata, "arXiv:2505.14689v3 [cs.AI] 21 May 2025"),
            2025,
        )
        self.assertEqual(_extract_year(metadata, "arXiv:2505.10303v1"), 2025)

    def test_abstract_removes_leading_period_and_line_wrap_artifacts(self):
        text = (
            "Abstract\n. Alternating automata allow verifica-\n tion and "
            "non-\n deterministic behavior.\n1 Introduction"
        )

        self.assertEqual(
            _extract_abstract(text),
            "Alternating automata allow verification and non-deterministic behavior.",
        )

    def test_latex_omega_in_title_becomes_unicode(self):
        self.assertEqual(
            _normalize_latex_symbols(r"Dynamic $\omega$-Regular Shielding"),
            "Dynamic ω-Regular Shielding",
        )

    def test_extract_pdf_pages_keeps_page_numbers(self):
        document = fitz.open()
        document.new_page().insert_text((72, 72), "First page finding")
        document.new_page().insert_text((72, 72), "Second page finding")
        pdf_data = document.tobytes()
        document.close()

        pages = extract_pdf_pages_bytes(pdf_data)

        self.assertEqual([page.page_number for page in pages], [1, 2])
        self.assertIn("First page finding", pages[0].text)
        self.assertIn("Second page finding", pages[1].text)

    def test_blank_pdf_has_no_preparable_text(self):
        document = fitz.open()
        document.new_page()
        pdf_data = document.tobytes()
        document.close()

        with self.assertRaisesRegex(PDFProcessingError, "No selectable text"):
            extract_pdf_pages_bytes(pdf_data)

    def test_corrupt_pdf_upload_is_removed_after_processing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            upload = FileStorage(
                stream=BytesIO(b"%PDF-1.7\nnot a real PDF"),
                filename="paper.pdf",
            )

            with self.assertRaisesRegex(PDFProcessingError, "not a readable PDF"):
                process_uploaded_pdf(upload, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
