import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

import fitz

from app import create_app
from app.db import get_db


def make_pdf(*page_texts):
    document = fitz.open()
    for text in page_texts:
        document.new_page().insert_text((72, 72), text)
    pdf_data = document.tobytes()
    document.close()
    return pdf_data


class PaperStorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.config = {
            "TESTING": True,
            "DATABASE": str(root / "library.sqlite3"),
            "UPLOAD_FOLDER": str(root / "uploads"),
            "GEMINI_MODEL": "test-model",
        }
        self.app = create_app(self.config)
        self.client = self.app.test_client()

    def save_arxiv_paper(self, external_id):
        response = self.client.post(
            "/library/save",
            data={
                "external_id": external_id,
                "title": "Persistent paper",
                "authors": "Test Author",
                "publication_year": "2025",
                "abstract": "A stored abstract.",
                "paper_url": f"https://arxiv.org/abs/{external_id}",
            },
        )
        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            return get_db().execute(
                "SELECT id FROM papers WHERE external_id = ?", (external_id,)
            ).fetchone()["id"]

    def test_saved_paper_remains_in_library_after_new_app_uses_same_database(self):
        paper_id = self.save_arxiv_paper("2505.12345")

        restarted_app = create_app(self.config)
        response = restarted_app.test_client().get("/library")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Persistent paper", response.data)
        self.assertIn(b"2505.12345", response.data)
        with restarted_app.app_context():
            rows = get_db().execute(
                "SELECT id, title, authors, publication_year, abstract "
                "FROM papers WHERE external_id = ?",
                ("2505.12345",),
            ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], paper_id)
        self.assertEqual(rows[0]["title"], "Persistent paper")
        self.assertEqual(rows[0]["authors"], "Test Author")
        self.assertEqual(rows[0]["publication_year"], 2025)
        self.assertEqual(rows[0]["abstract"], "A stored abstract.")

    def test_repreparing_replaces_old_chunks_and_invalidates_summary(self):
        external_id = "2505.67890"
        paper_id = self.save_arxiv_paper(external_id)
        old_pdf = make_pdf("Old first finding.", "Old second finding.")
        new_pdf = make_pdf("New replacement finding.")

        with patch(
            "app.services.paper_preparation.download_paper_pdf",
            side_effect=[old_pdf, new_pdf],
        ) as download:
            first = self.client.post(
                f"/papers/{paper_id}/prepare", follow_redirects=True
            )
            self.assertEqual(first.status_code, 200)
            self.assertIn(b"2 text chunks saved", first.data)

            with patch(
                "app.generate_paper_summary", return_value="Summary of old findings."
            ) as summarize:
                summarized = self.client.post(
                    f"/papers/{paper_id}/summary", follow_redirects=True
                )
            self.assertEqual(summarized.status_code, 200)
            summarize.assert_called_once()

            with self.app.app_context():
                database = get_db()
                old_chunks = database.execute(
                    "SELECT page_number, content FROM paper_chunks "
                    "WHERE paper_id = ? ORDER BY chunk_index",
                    (paper_id,),
                ).fetchall()
                old_summary = database.execute(
                    "SELECT summary FROM paper_summaries WHERE paper_id = ?",
                    (paper_id,),
                ).fetchone()
            self.assertEqual([row["page_number"] for row in old_chunks], [1, 2])
            self.assertIn("Old first finding.", old_chunks[0]["content"])
            self.assertIn("Old second finding.", old_chunks[1]["content"])
            self.assertEqual(old_summary["summary"], "Summary of old findings.")

            second = self.client.post(
                f"/papers/{paper_id}/prepare", follow_redirects=True
            )

        self.assertEqual(second.status_code, 200)
        self.assertIn(b"1 text chunks saved", second.data)
        self.assertNotIn(b"Summary of old findings.", second.data)
        self.assertEqual(download.call_args_list, [call(external_id), call(external_id)])
        with self.app.app_context():
            database = get_db()
            new_chunks = database.execute(
                "SELECT chunk_index, page_number, content FROM paper_chunks "
                "WHERE paper_id = ? ORDER BY chunk_index",
                (paper_id,),
            ).fetchall()
            old_summary = database.execute(
                "SELECT summary FROM paper_summaries WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
        self.assertEqual(len(new_chunks), 1)
        self.assertEqual(new_chunks[0]["chunk_index"], 0)
        self.assertEqual(new_chunks[0]["page_number"], 1)
        self.assertIn("New replacement finding.", new_chunks[0]["content"])
        self.assertNotIn("Old", new_chunks[0]["content"])
        self.assertIsNone(old_summary)


if __name__ == "__main__":
    unittest.main()
