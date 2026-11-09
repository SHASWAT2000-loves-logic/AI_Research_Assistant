import random
import re
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import fitz

from app import create_app
from app.db import get_db
from app.services.arxiv import ArxivError
from app.services.llm import PaperAIError
from app.services.paper_preparation import PaperPreparationError


def make_upload_pdf(*, attachment_size=0):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Valid uploaded paper", fontsize=18)
    page.insert_text((72, 105), "Test Author")
    page.insert_text((72, 140), "Abstract")
    page.insert_text((72, 165), "This paper has selectable text.")
    page.insert_text((72, 200), "1 Introduction")
    if attachment_size:
        document.embfile_add(
            "supplement.bin", random.Random(0).randbytes(attachment_size)
        )
    pdf_data = document.tobytes()
    document.close()
    return pdf_data


class AppErrorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.upload_folder = root / "uploads"
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": str(root / "test.sqlite3"),
                "UPLOAD_FOLDER": str(self.upload_folder),
                "GEMINI_MODEL": "test-model",
            }
        )
        self.client = self.app.test_client()

    def add_paper(self, *, prepared=False):
        with self.app.app_context():
            database = get_db()
            paper_id = database.execute(
                "INSERT INTO papers (title, source) VALUES (?, ?)",
                ("Test paper", "upload"),
            ).lastrowid
            if prepared:
                database.execute(
                    "INSERT INTO paper_chunks (paper_id, chunk_index, page_number, content) "
                    "VALUES (?, ?, ?, ?)",
                    (paper_id, 0, 1, "This paper evaluates 42 participants."),
                )
            database.commit()
        return paper_id

    @patch("app.search_papers")
    def test_blank_search_shows_validation_without_external_request(self, search):
        response = self.client.get("/search?query=%20%20")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Enter a research topic or keyword", response.data)
        search.assert_not_called()

    @patch("app.search_papers", side_effect=ArxivError("offline"))
    def test_search_failure_shows_recoverable_error(self, search):
        response = self.client.get("/search?query=%20graph%20")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"arXiv search is temporarily unavailable", response.data)
        search.assert_called_once_with("graph")

    def test_duplicate_arxiv_save_keeps_one_database_record(self):
        data = {
            "external_id": "2505.12345",
            "title": "Example paper",
            "publication_year": "not-a-year",
        }

        first = self.client.post("/library/save", data=data, follow_redirects=True)
        second = self.client.post("/library/save", data=data, follow_redirects=True)

        self.assertIn(b"Paper saved to your library", first.data)
        self.assertIn(b"already in your library", second.data)
        with self.app.app_context():
            rows = get_db().execute(
                "SELECT title, publication_year FROM papers WHERE external_id = ?",
                ("2505.12345",),
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "Example paper")
        self.assertIsNone(rows[0]["publication_year"])

    def test_save_requires_identifier_and_title(self):
        for data in ({"title": "A title"}, {"external_id": "2505.12345"}):
            with self.subTest(data=data):
                response = self.client.post("/library/save", data=data)
                self.assertEqual(response.status_code, 400)

        with self.app.app_context():
            self.assertEqual(get_db().execute("SELECT COUNT(*) FROM papers").fetchone()[0], 0)

    def test_upload_rejects_non_pdf_bytes_without_storing_file(self):
        response = self.client.post(
            "/upload",
            data={"paper": (BytesIO(b"not a PDF"), "paper.pdf")},
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(b"not a valid PDF", response.data)
        self.assertEqual(list(self.upload_folder.iterdir()), [])

    def test_upload_over_size_limit_returns_413(self):
        limit = 1024 * 1024
        self.app.config["MAX_CONTENT_LENGTH"] = limit
        small_pdf = make_upload_pdf()
        large_pdf = make_upload_pdf(attachment_size=1_200_000)
        self.assertLess(len(small_pdf), limit)
        self.assertGreater(len(large_pdf), limit)

        accepted = self.client.post(
            "/upload",
            data={"paper": (BytesIO(small_pdf), "small.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(accepted.status_code, 200)
        self.assertIn(b"Check the extracted details", accepted.data)
        match = re.search(rb'name="stored_filename" value="([^"]+)"', accepted.data)
        self.assertIsNotNone(match)
        stored_filename = match.group(1).decode()
        self.assertEqual((self.upload_folder / stored_filename).read_bytes(), small_pdf)
        with self.app.app_context():
            self.assertEqual(get_db().execute("SELECT COUNT(*) FROM papers").fetchone()[0], 0)

        confirmed = self.client.post(
            "/upload/confirm",
            data={"stored_filename": stored_filename, "title": "Small valid paper"},
            follow_redirects=True,
        )
        self.assertEqual(confirmed.status_code, 200)
        with self.app.app_context():
            saved = get_db().execute(
                "SELECT title, pdf_path FROM papers WHERE pdf_path = ?",
                (stored_filename,),
            ).fetchall()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["title"], "Small valid paper")

        rejected = self.client.post(
            "/upload",
            data={"paper": (BytesIO(large_pdf), "large.pdf")},
            content_type="multipart/form-data",
        )
        self.addCleanup(rejected.request.environ["wsgi.input"].close)

        self.assertEqual(rejected.status_code, 413)
        self.assertIn(b"The PDF is too large", rejected.data)
        self.assertEqual(
            [path.name for path in self.upload_folder.iterdir()], [stored_filename]
        )
        self.assertEqual((self.upload_folder / stored_filename).read_bytes(), small_pdf)
        with self.app.app_context():
            remaining = get_db().execute(
                "SELECT title, pdf_path FROM papers"
            ).fetchall()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["title"], "Small valid paper")
        self.assertEqual(remaining[0]["pdf_path"], stored_filename)

    def test_upload_confirmation_rejects_path_escape(self):
        response = self.client.post(
            "/upload/confirm",
            data={"title": "Unsafe paper", "stored_filename": "../paper.pdf"},
        )

        self.assertEqual(response.status_code, 400)
        with self.app.app_context():
            self.assertEqual(get_db().execute("SELECT COUNT(*) FROM papers").fetchone()[0], 0)

    @patch("app.prepare_paper", side_effect=PaperPreparationError("PDF unavailable"))
    def test_prepare_failure_does_not_store_chunks(self, prepare):
        paper_id = self.add_paper()

        response = self.client.post(f"/papers/{paper_id}/prepare")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"PDF unavailable", response.data)
        prepare.assert_called_once()
        with self.app.app_context():
            count = get_db().execute(
                "SELECT COUNT(*) FROM paper_chunks WHERE paper_id = ?", (paper_id,)
            ).fetchone()[0]
        self.assertEqual(count, 0)

    @patch("app.generate_paper_summary", side_effect=PaperAIError("Gemini unavailable"))
    def test_summary_failure_does_not_save_summary(self, summarize):
        paper_id = self.add_paper(prepared=True)

        response = self.client.post(f"/papers/{paper_id}/summary")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Gemini unavailable", response.data)
        summarize.assert_called_once()
        with self.app.app_context():
            count = get_db().execute(
                "SELECT COUNT(*) FROM paper_summaries WHERE paper_id = ?", (paper_id,)
            ).fetchone()[0]
        self.assertEqual(count, 0)

    @patch("app.answer_paper_question")
    def test_overlong_question_does_not_call_gemini(self, answer):
        paper_id = self.add_paper(prepared=True)

        response = self.client.post(
            f"/papers/{paper_id}/ask", data={"question": "x" * 1001}
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Keep the question under 1,000 characters", response.data)
        answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
