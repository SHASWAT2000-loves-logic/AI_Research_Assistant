from .db import get_db


def save_arxiv_paper(
    *, external_id, title, authors, publication_year, abstract, paper_url
):
    database = get_db()
    cursor = database.execute(
        """
        INSERT OR IGNORE INTO papers (
            title,
            authors,
            publication_year,
            abstract,
            paper_url,
            source,
            external_id
        )
        VALUES (?, ?, ?, ?, ?, 'arxiv', ?)
        """,
        (
            title,
            authors,
            publication_year,
            abstract,
            paper_url,
            external_id,
        ),
    )
    database.commit()
    return cursor.rowcount == 1


def save_uploaded_paper(
    *, title, authors, publication_year, abstract, stored_filename
):
    database = get_db()
    cursor = database.execute(
        """
        INSERT INTO papers (
            title,
            authors,
            publication_year,
            abstract,
            source,
            pdf_path
        )
        VALUES (?, ?, ?, ?, 'upload', ?)
        """,
        (title, authors, publication_year, abstract, stored_filename),
    )
    database.commit()
    return cursor.lastrowid


def list_papers():
    return get_db().execute(
        """
        SELECT id, title, authors, publication_year, abstract, source, external_id
        FROM papers
        ORDER BY created_at DESC, id DESC
        """
    ).fetchall()


def get_paper(paper_id):
    return get_db().execute(
        """
        SELECT id, title, authors, publication_year, abstract, paper_url,
               source, external_id, pdf_path, created_at,
               (
                   SELECT COUNT(*)
                   FROM paper_chunks
                   WHERE paper_chunks.paper_id = papers.id
               ) AS chunk_count,
               (
                   SELECT MAX(created_at)
                   FROM paper_chunks
                   WHERE paper_chunks.paper_id = papers.id
               ) AS prepared_at
        FROM papers
        WHERE id = ?
        """,
        (paper_id,),
    ).fetchone()


def replace_paper_chunks(paper_id, chunks):
    database = get_db()
    with database:
        database.execute("DELETE FROM paper_summaries WHERE paper_id = ?", (paper_id,))
        database.execute("DELETE FROM paper_chunks WHERE paper_id = ?", (paper_id,))
        database.executemany(
            """
            INSERT INTO paper_chunks (
                paper_id,
                chunk_index,
                page_number,
                content
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                (paper_id, chunk.chunk_index, chunk.page_number, chunk.content)
                for chunk in chunks
            ),
        )
    return len(chunks)


def get_paper_chunks(paper_id):
    return get_db().execute(
        """
        SELECT chunk_index, page_number, content
        FROM paper_chunks
        WHERE paper_id = ?
        ORDER BY chunk_index
        """,
        (paper_id,),
    ).fetchall()


def get_paper_summary(paper_id):
    return get_db().execute(
        """
        SELECT summary, model, created_at, updated_at
        FROM paper_summaries
        WHERE paper_id = ?
        """,
        (paper_id,),
    ).fetchone()


def save_paper_summary(*, paper_id, summary, model):
    database = get_db()
    database.execute(
        """
        INSERT INTO paper_summaries (paper_id, summary, model)
        VALUES (?, ?, ?)
        ON CONFLICT(paper_id) DO UPDATE SET
            summary = excluded.summary,
            model = excluded.model,
            updated_at = CURRENT_TIMESTAMP
        """,
        (paper_id, summary, model),
    )
    database.commit()
