# Short demo plan

Before recording, start the app with `GEMINI_API_KEY` set. Choose one arXiv result that is not already in the Library and one text-based PDF with a question you can check against the paper. Keep the API key off screen.

1. **Search:** Enter a research topic on Search. Show a result's title, authors, year, abstract, and paper link.
2. **Save:** Save that result to the Library. Open Library and refresh once to show it remains saved.
3. **Upload:** Upload the PDF. Show the extracted metadata review page, correct a field if needed, and save it. Open the uploaded paper from Library.
4. **Summarize:** Click **Prepare paper for AI**, show the saved chunk count, then click **Generate AI summary**. Point to a finding in the summary that appears in the PDF.
5. **Ask:** Ask a question whose answer is in the PDF. Show the answer and its page citation, then briefly check that page in the PDF.

This covers the five actions the assignment requires in the video. Summary and question generation each use one Gemini request.
