import os
import unittest
from unittest.mock import patch

import httpx
from google.genai import errors

from app.services.llm import PaperAIError, _request_gemini


class GeminiRequestTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
        environment.start()
        self.addCleanup(environment.stop)

        client_patch = patch("app.services.llm.genai.Client")
        self.client_class = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.create = self.client_class.return_value.interactions.create

    def request(self):
        return _request_gemini(
            prompt="Prepared paper text and question",
            model="test-model",
            empty_response_message="Gemini returned no answer.",
        )

    def test_missing_api_key_stops_before_creating_client(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "   "}):
            with self.assertRaisesRegex(PaperAIError, "Set the GEMINI_API_KEY"):
                self.request()

        self.client_class.assert_not_called()

    def test_success_sends_model_prompt_and_store_false_and_returns_text(self):
        self.create.return_value.output_text = "  Supported answer.  \n"

        result = self.request()

        self.assertEqual(result, "Supported answer.")
        self.client_class.assert_called_once_with(api_key="test-key")
        self.create.assert_called_once_with(
            model="test-model", input="Prepared paper text and question", store=False
        )

    def test_empty_output_raises_paper_ai_error(self):
        self.create.return_value.output_text = " \n "

        with self.assertRaisesRegex(PaperAIError, "Gemini returned no answer"):
            self.request()
        self.create.assert_called_once()

    def test_invalid_access_reports_key_or_project_problem(self):
        error = errors.APIError(
            401, {"status": "UNAUTHENTICATED", "message": "Invalid API key"}
        )
        self.create.side_effect = error

        with self.assertRaises(PaperAIError) as raised:
            self.request()

        self.assertIn("API key or project access", str(raised.exception))
        self.assertIn("GEMINI_API_KEY", str(raised.exception))
        self.assertIs(raised.exception.__cause__, error)

    def test_unavailable_model_names_configured_model(self):
        error = errors.APIError(
            404, {"status": "NOT_FOUND", "message": "Model not found"}
        )
        self.create.side_effect = error

        with self.assertRaises(PaperAIError) as raised:
            self.request()

        self.assertIn('configured Gemini model "test-model" is not available', str(raised.exception))
        self.assertIn("GEMINI_MODEL", str(raised.exception))
        self.assertIs(raised.exception.__cause__, error)

    def test_daily_quota_error_gives_quota_guidance(self):
        error = errors.APIError(
            429, {"status": "RESOURCE_EXHAUSTED", "message": "Daily quota exceeded"}
        )
        self.create.side_effect = error

        with self.assertRaises(PaperAIError) as raised:
            self.request()

        self.assertIn("free-tier quota has been used", str(raised.exception))
        self.assertIn("Google AI Studio", str(raised.exception))
        self.assertIs(raised.exception.__cause__, error)

    def test_rate_limit_error_asks_user_to_retry(self):
        error = errors.APIError(
            429, {"status": "RESOURCE_EXHAUSTED", "message": "Too many requests"}
        )
        self.create.side_effect = error

        with self.assertRaises(PaperAIError) as raised:
            self.request()

        self.assertIn("rate limit was reached", str(raised.exception))
        self.assertIn("try again", str(raised.exception))
        self.assertIs(raised.exception.__cause__, error)

    def test_server_failure_reports_temporary_unavailability(self):
        error = errors.APIError(
            503, {"status": "UNAVAILABLE", "message": "Service unavailable"}
        )
        self.create.side_effect = error

        with self.assertRaises(PaperAIError) as raised:
            self.request()

        self.assertIn("Gemini is temporarily unavailable", str(raised.exception))
        self.assertIs(raised.exception.__cause__, error)

    def test_httpx_connect_error_becomes_paper_ai_error(self):
        error = httpx.ConnectError("connection refused")
        self.create.side_effect = error

        with self.assertRaisesRegex(PaperAIError, "could not connect to Gemini") as raised:
            self.request()

        self.assertIs(raised.exception.__cause__, error)

    def test_httpx_timeout_becomes_paper_ai_error(self):
        error = httpx.TimeoutException("request timed out")
        self.create.side_effect = error

        with self.assertRaisesRegex(PaperAIError, "could not connect to Gemini") as raised:
            self.request()

        self.assertIs(raised.exception.__cause__, error)


if __name__ == "__main__":
    unittest.main()
