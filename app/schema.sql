CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    authors TEXT,
    publication_year INTEGER,
    abstract TEXT,
    paper_url TEXT,
    source TEXT NOT NULL DEFAULT 'local',
    external_id TEXT,
    pdf_path TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_source_external_id
ON papers (source, external_id)
WHERE external_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS paper_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE,
    UNIQUE (paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_paper_chunks_paper_id
ON paper_chunks (paper_id, chunk_index);

CREATE TABLE IF NOT EXISTS paper_summaries (
    paper_id INTEGER PRIMARY KEY,
    summary TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE
);
