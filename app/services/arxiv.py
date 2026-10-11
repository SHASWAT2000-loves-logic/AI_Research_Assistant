from dataclasses import dataclass
from threading import Lock
from time import monotonic, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree


API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"
REQUEST_INTERVAL_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESULTS = 10

_request_lock = Lock()
_last_request_started = 0.0


class ArxivError(Exception):
    """Raised when a paper search cannot be completed."""


@dataclass(frozen=True)
class PaperSearchResult:
    arxiv_id: str
    title: str
    authors: tuple[str, ...]
    publication_year: int | None
    abstract: str
    abstract_url: str
    pdf_url: str


def _clean_text(value):
    return " ".join((value or "").split())


def _parse_feed(feed):
    try:
        root = ElementTree.fromstring(feed)
    except ElementTree.ParseError as error:
        raise ArxivError("arXiv returned an invalid response.") from error

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []

    for entry in root.findall("atom:entry", namespace):
        entry_url = _clean_text(entry.findtext("atom:id", default="", namespaces=namespace))
        published = _clean_text(
            entry.findtext("atom:published", default="", namespaces=namespace)
        )
        authors = tuple(
            _clean_text(author.findtext("atom:name", default="", namespaces=namespace))
            for author in entry.findall("atom:author", namespace)
        )
        authors = tuple(author for author in authors if author)

        abstract_url = entry_url
        pdf_url = ""
        for link in entry.findall("atom:link", namespace):
            if link.get("rel") == "alternate":
                abstract_url = link.get("href", abstract_url)
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")

        try:
            publication_year = int(published[:4])
        except ValueError:
            publication_year = None

        papers.append(
            PaperSearchResult(
                arxiv_id=entry_url.rsplit("/", 1)[-1],
                title=_clean_text(
                    entry.findtext("atom:title", default="Untitled paper", namespaces=namespace)
                ),
                authors=authors,
                publication_year=publication_year,
                abstract=_clean_text(
                    entry.findtext("atom:summary", default="", namespaces=namespace)
                ),
                abstract_url=abstract_url,
                pdf_url=pdf_url,
            )
        )

    return papers


def _fetch(url):
    global _last_request_started

    request = Request(
        url,
        headers={"User-Agent": "AIResearchAssistant/1.0 (course project)"},
    )

    with _request_lock:
        wait_time = REQUEST_INTERVAL_SECONDS - (monotonic() - _last_request_started)
        if wait_time > 0:
            sleep(wait_time)

        _last_request_started = monotonic()
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise ArxivError("The arXiv request failed.") from error


def search_papers(query, max_results=MAX_RESULTS):
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    parameters = urlencode(
        {
            "search_query": f"all:{cleaned_query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
    )
    return _parse_feed(_fetch(f"{API_URL}?{parameters}"))


def download_paper_pdf(arxiv_id):
    cleaned_id = arxiv_id.strip()
    if not cleaned_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/" for character in cleaned_id):
        raise ArxivError("The saved arXiv identifier is invalid.")

    encoded_id = quote(cleaned_id, safe="/.-")
    return _fetch(PDF_URL.format(arxiv_id=encoded_id))
