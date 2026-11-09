import unittest
from urllib.error import URLError
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from app.services import arxiv


ATOM_FEED = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://arxiv.org/abs/2505.12345</id>
    <published>2025-05-19T00:00:00Z</published>
    <title>  Graph\n  Methods  </title>
    <summary>  A useful\n result.  </summary>
    <author><name>Ada Example</name></author>
    <author><name>Ben Researcher</name></author>
    <link rel="alternate" href="https://arxiv.org/abs/2505.12345" />
    <link title="pdf" type="application/pdf"
          href="https://arxiv.org/pdf/2505.12345" />
  </entry>
</feed>"""


class ArxivServiceTests(unittest.TestCase):
    def test_parse_feed_extracts_paper_metadata(self):
        papers = arxiv._parse_feed(ATOM_FEED)

        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.arxiv_id, "2505.12345")
        self.assertEqual(paper.title, "Graph Methods")
        self.assertEqual(paper.authors, ("Ada Example", "Ben Researcher"))
        self.assertEqual(paper.publication_year, 2025)
        self.assertEqual(paper.abstract, "A useful result.")
        self.assertEqual(paper.abstract_url, "https://arxiv.org/abs/2505.12345")
        self.assertEqual(paper.pdf_url, "https://arxiv.org/pdf/2505.12345")

    def test_invalid_feed_raises_arxiv_error(self):
        with self.assertRaisesRegex(arxiv.ArxivError, "invalid response"):
            arxiv._parse_feed(b"<feed><entry></feed>")

    @patch("app.services.arxiv._fetch", return_value=ATOM_FEED)
    def test_search_trims_query_and_sends_requested_result_limit(self, fetch):
        papers = arxiv.search_papers("  graph learning  ", max_results=3)

        self.assertEqual(len(papers), 1)
        url = urlsplit(fetch.call_args.args[0])
        self.assertEqual(url.scheme, "https")
        self.assertEqual(url.netloc, "export.arxiv.org")
        self.assertEqual(
            parse_qs(url.query)["search_query"], ["all:graph learning"]
        )
        self.assertEqual(parse_qs(url.query)["max_results"], ["3"])

    @patch("app.services.arxiv._fetch")
    def test_blank_search_does_not_request_arxiv(self, fetch):
        self.assertEqual(arxiv.search_papers(" \t "), [])
        fetch.assert_not_called()

    def test_fetch_waits_for_three_second_spacing(self):
        with (
            patch.object(arxiv, "_last_request_started", 9.0),
            patch.object(arxiv, "monotonic", side_effect=[10.0, 12.0]),
            patch.object(arxiv, "sleep") as sleep,
            patch.object(arxiv, "urlopen") as open_url,
        ):
            open_url.return_value.__enter__.return_value.read.return_value = b"feed"

            self.assertEqual(arxiv._fetch("https://example.org/feed"), b"feed")
            sleep.assert_called_once_with(2.0)
            self.assertEqual(arxiv._last_request_started, 12.0)
            request = open_url.call_args.args[0]
            self.assertEqual(request.full_url, "https://example.org/feed")
            self.assertTrue(request.get_header("User-agent"))
            self.assertEqual(open_url.call_args.kwargs["timeout"], 20)

    def test_fetch_converts_network_failure_to_arxiv_error(self):
        with (
            patch.object(arxiv, "_last_request_started", 0.0),
            patch.object(arxiv, "monotonic", return_value=100.0),
            patch.object(arxiv, "urlopen", side_effect=URLError("offline")),
        ):
            with self.assertRaisesRegex(arxiv.ArxivError, "request failed"):
                arxiv._fetch("https://example.org/feed")

    @patch("app.services.arxiv._fetch")
    def test_invalid_pdf_identifier_does_not_request_arxiv(self, fetch):
        with self.assertRaisesRegex(arxiv.ArxivError, "identifier is invalid"):
            arxiv.download_paper_pdf("2505.12345?download=true")
        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
