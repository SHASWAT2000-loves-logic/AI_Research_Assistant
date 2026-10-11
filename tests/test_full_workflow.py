import re
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import fitz

from app import create_app
from app.db import get_db
from app.services.arxiv import PaperSearchResult


def make_pdf(title, finding):
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), title, fontsize=18)
    page.insert_text((72, 105), "Demo Author", fontsize=11)
    page.insert_text((72, 150), "Abstract", fontsize=12)
    page.insert_text((72, 170), finding, fontsize=11)
    page.insert_text((72, 205), "1 Introduction", fontsize=12)
    page.insert_text((72, 225), finding, fontsize=11)
    document.set_metadata({"title": title, "author": "Demo Author"})
    pdf_data = document.tobytes()
    document.close()
    return pdf_data


class FullWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": str(root / "demo.sqlite3"),
                "UPLOAD_FOLDER": str(root / "uploads"),
                "GEMINI_MODEL": "test-model",
            }
        )
        self.client = self.app.test_client()

    def test_search_save_upload_prepare_summarize_and_ask(self):
        arxiv_finding = "The search paper evaluates eighty samples."
        upload_finding = "The uploaded paper evaluates forty-two participants."
        arxiv_pdf = make_pdf("Search Paper", arxiv_finding)
        upload_pdf = make_pdf("Uploaded Paper", upload_finding)
        result = PaperSearchResult(
            arxiv_id="2601.12345",
            title="Search Paper",
            authors=("Demo Author",),
            publication_year=2026,
            abstract=arxiv_finding,
            abstract_url="https://arxiv.org/abs/2601.12345",
            pdf_url="https://arxiv.org/pdf/2601.12345",
        )

        with patch("app.search_papers", return_value=[result]) as search:
            response = self.client.get("/search?query=demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Search Paper", response.data)
        search.assert_called_once_with("demo")

        response = self.client.post(
            "/library/save",
            data={
                "external_id": result.arxiv_id,
                "title": result.title,
                "authors": ", ".join(result.authors),
                "publication_year": result.publication_year,
                "abstract": result.abstract,
                "paper_url": result.abstract_url,
            },
            follow_redirects=True,
        )
        self.assertIn(b"Paper saved to your library.", response.data)
        with self.app.app_context():
            arxiv_id = get_db().execute(
                "SELECT id FROM papers WHERE external_id = ?", (result.arxiv_id,)
            ).fetchone()["id"]

        with patch(
            "app.services.paper_preparation.download_paper_pdf",
            return_value=arxiv_pdf,
        ) as download:
            response = self.client.post(
                f"/papers/{arxiv_id}/prepare", follow_redirects=True
            )
        self.assertIn(b"Paper prepared successfully", response.data)
        download.assert_called_once_with(result.arxiv_id)

        response = self.client.post(
            "/upload",
            data={"paper": (BytesIO(upload_pdf), "uploaded-paper.pdf")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Check the extracted details", response.data)
        stored_filename = re.search(
            rb'name="stored_filename" value="([^"]+)"', response.data
        ).group(1).decode()

        response = self.client.post(
            "/upload/confirm",
            data={
                "stored_filename": stored_filename,
                "title": "Uploaded Paper",
                "authors": "Demo Author",
                "publication_year": "2026",
                "abstract": upload_finding,
            },
            follow_redirects=True,
        )
        self.assertIn(b"Uploaded Paper", response.data)
        with self.app.app_context():
            uploaded_id = get_db().execute(
                "SELECT id FROM papers WHERE pdf_path = ?", (stored_filename,)
            ).fetchone()["id"]

        response = self.client.post(
            f"/papers/{uploaded_id}/prepare", follow_redirects=True
        )
        self.assertIn(b"Paper prepared successfully", response.data)
        with self.app.app_context():
            database = get_db()
            self.assertGreater(
                database.execute(
                    "SELECT COUNT(*) FROM paper_chunks WHERE paper_id = ?",
                    (uploaded_id,),
                ).fetchone()[0],
                0,
            )

        summary = "Overview\nThe uploaded paper evaluates forty-two participants."
        answer = "It evaluates forty-two participants (p. 1)."
        with patch("app.services.llm._request_gemini", side_effect=[summary, answer]) as ai:
            response = self.client.post(
                f"/papers/{uploaded_id}/summary", follow_redirects=True
            )
            self.assertIn(summary.encode(), response.data)
            response = self.client.post(
                f"/papers/{uploaded_id}/ask",
                data={"question": "How many participants were evaluated?"},
            )
        self.assertIn(answer.encode(), response.data)
        self.assertEqual(ai.call_count, 2)
        self.assertIn(upload_finding, ai.call_args_list[0].kwargs["prompt"])
        self.assertIn(upload_finding, ai.call_args_list[1].kwargs["prompt"])
        self.assertNotIn(arxiv_finding, ai.call_args_list[1].kwargs["prompt"])
        with self.app.app_context():
            saved_summary = get_db().execute(
                "SELECT summary FROM paper_summaries WHERE paper_id = ?",
                (uploaded_id,),
            ).fetchone()["summary"]
        self.assertEqual(saved_summary, summary)


if __name__ == "__main__":
    unittest.main()
