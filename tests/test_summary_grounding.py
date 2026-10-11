import unittest
from unittest.mock import patch

from app.services.llm import PaperAIError, generate_paper_summary


def summary(limitation):
    return (
        "Overview\nA study of the subject.\n"
        "Main contributions\nIt reports a result.\n"
        "Methods and evidence\nThe authors analyze their data.\n"
        f"Limitations\n- {limitation}"
    )


class SummaryGroundingTests(unittest.TestCase):
    @patch("app.services.llm._request_gemini")
    def test_one_request_for_a_paper_with_a_negative_result(self, request):
        source = (
            "The broad claim of a benefit for all ages is false. We observed a "
            "benefit only for participants aged 65 and older. Whether this result "
            "extends to adults under 65 remains open."
        )
        corrected = summary(
            "A benefit for all ages is false; it was observed only for adults "
            "aged 65 and older. Whether it extends to adults under 65 remains open."
        )
        request.return_value = corrected

        result = generate_paper_summary(
            title="Example study",
            authors="Example author",
            chunks=[{"page_number": 4, "chunk_index": 0, "content": source}],
            model="test",
        )

        self.assertEqual(result, corrected)
        self.assertEqual(request.call_count, 1)
        self.assertIn(source, request.call_args.kwargs["prompt"])
        self.assertIn("explicit negative results", request.call_args.kwargs["prompt"])

    @patch("app.services.llm._request_gemini")
    def test_conclusion_is_prioritized_with_one_request(self, request):
        source = (
            "The unreserved statement certainly does not hold, even for "
            "fixeddepth GNNs. What if we relativize more generally to MSO "
            "(monadic second-order logic) node classifiers and strongly "
            "connected graphs? The general case where graphs are not "
            "necessarily strongly connected remains wide open."
        )
        corrected = summary(
            "The unrestricted converse is false even for fixed-depth GNNs. "
            "The question concerning MSO-expressible node classifiers on "
            "strongly connected graphs remains open; the general case of "
            "MSO-expressible classifiers on other graphs is also open."
        )
        request.return_value = corrected

        result = generate_paper_summary(
            title="Example graph paper",
            authors="Example author",
            chunks=[
                {"page_number": 1, "chunk_index": 0, "content": "Introduction text."},
                {"page_number": 9, "chunk_index": 19, "content": "6 Concluding remarks " + source},
                {"page_number": 9, "chunk_index": 20, "content": "Further discussion."},
                {"page_number": 10, "chunk_index": 21, "content": "Appendix calculations."},
            ],
            model="test",
        )

        self.assertEqual(result, corrected)
        self.assertEqual(request.call_count, 1)
        prompt = request.call_args.kwargs["prompt"]
        self.assertLess(prompt.index("Appendix calculations."), prompt.index(source))

    @patch("app.services.llm._request_gemini")
    def test_one_request_for_a_paper_without_a_conclusion_heading(self, request):
        draft = summary("The sample size was small.")
        request.return_value = draft

        result = generate_paper_summary(
            title="Another study",
            authors="Another author",
            chunks=[
                {
                    "page_number": 2,
                    "chunk_index": 1,
                    "content": "The sample size was small.",
                }
            ],
            model="test",
        )

        self.assertEqual(result, draft)
        self.assertEqual(request.call_count, 1)

    @patch("app.services.llm._request_gemini")
    def test_unprepared_paper_makes_no_request(self, request):
        with self.assertRaises(PaperAIError):
            generate_paper_summary(
                title="Another study",
                authors="Another author",
                chunks=[],
                model="test",
            )
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
