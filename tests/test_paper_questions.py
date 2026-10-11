import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import create_app
from app.db import get_db


class PaperQuestionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.app = create_app(
            {
                "TESTING": True,
                "DATABASE": str(root / "test.sqlite3"),
                "UPLOAD_FOLDER": str(root / "uploads"),
                "GEMINI_MODEL": "test-model",
            }
        )
        self.client = self.app.test_client()
        with self.app.app_context():
            database = get_db()
            self.prepared_id = database.execute(
                "INSERT INTO papers (title, authors, source) VALUES (?, ?, ?)",
                ("Example paper", "Researcher", "upload"),
            ).lastrowid
            database.execute(
                "INSERT INTO paper_chunks (paper_id, chunk_index, page_number, content) "
                "VALUES (?, ?, ?, ?)",
                (self.prepared_id, 0, 2, "The experiment used 42 participants."),
            )
            self.unprepared_id = database.execute(
                "INSERT INTO papers (title, authors, source) VALUES (?, ?, ?)",
                ("Unprepared paper", "Researcher", "upload"),
            ).lastrowid
            database.commit()

    @patch("app.services.llm._request_gemini", return_value="It used 42 participants (p. 2).")
    def test_prepared_paper_answer_uses_its_page_numbered_text(self, request):
        response = self.client.post(
            f"/papers/{self.prepared_id}/ask",
            data={"question": "How many participants were used?"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"It used 42 participants (p. 2).", response.data)
        self.assertIn(b"How many participants were used?", response.data)
        request.assert_called_once()
        prompt = request.call_args.kwargs["prompt"]
        self.assertIn("[Page 2, chunk 1]", prompt)
        self.assertIn("The experiment used 42 participants.", prompt)
        self.assertIn("How many participants were used?", prompt)
        self.assertIn("Cite supporting page numbers", prompt)

    @patch("app.services.llm._request_gemini")
    def test_unprepared_paper_cannot_be_asked(self, request):
        page = self.client.get(f"/papers/{self.unprepared_id}")
        response = self.client.post(
            f"/papers/{self.unprepared_id}/ask",
            data={"question": "What did it find?"},
        )

        self.assertIn(b"disabled", page.data)
        self.assertIn(b"Prepare this paper before asking questions", response.data)
        request.assert_not_called()

    @patch("app.services.llm._request_gemini")
    def test_empty_question_does_not_call_gemini(self, request):
        response = self.client.post(
            f"/papers/{self.prepared_id}/ask", data={"question": "   "}
        )

        self.assertIn(b"Enter a question about this paper.", response.data)
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
