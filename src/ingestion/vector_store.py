import os
import sqlite3
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SQLiteVectorStore")

class SQLiteVectorStore:
    """
    Resilient Relational Vector Store Engine utilizing native SQLite.
    Fully enforces Parent-Child structural mapping to prevent text data fragmentation.
    """
    def __init__(self):
        self.db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/financial_intelligence.db"))
        self._initialize_database()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_database(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys = ON;")
            
            # Parent Document holds the unfragmented global structural view
            cur.execute("""
                CREATE TABLE IF NOT EXISTS parent_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_file TEXT NOT NULL,
                    section_title TEXT,
                    full_content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Child Chunks hold granular token frames for hyper-focused math-matching
            cur.execute("""
                CREATE TABLE IF NOT EXISTS child_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parent_id INTEGER,
                    chunk_content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_id) REFERENCES parent_documents(id) ON DELETE CASCADE
                );
            """)
            conn.commit()
        logger.info(f"⚡ Standalone Relational database initialized at: {self.db_path}")

    def insert_document_pipeline(self, source_file: str, section_title: str, parent_text: str, child_tuples: list):
        with self._get_connection() as conn:
            cur = conn.cursor()
            try:
                cur.execute(
                    "INSERT INTO parent_documents (source_file, section_title, full_content) VALUES (?, ?, ?);",
                    (source_file, section_title, parent_text)
                )
                parent_id = cur.lastrowid

                child_data = [
                    (parent_id, text, json.dumps(vector)) for text, vector in child_tuples
                ]
                cur.executemany(
                    "INSERT INTO child_chunks (parent_id, chunk_content, embedding_json) VALUES (?, ?, ?);",
                    child_data
                )
                conn.commit()
                logger.info(f"💾 Transaction Success: Ingested parent row [{parent_id}] with {len(child_tuples)} bound child chunks.")
            except Exception as e:
                conn.rollback()
                logger.error(f"❌ Storage execution rolled back: {str(e)}")
                raise e

    def get_ingested_source_files(self) -> set:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT source_file FROM parent_documents;")
            return {row[0] for row in cur.fetchall()}

    def fetch_all_child_chunks(self) -> list:
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT c.id, c.parent_id, c.chunk_content, c.embedding_json,
                       p.source_file, p.section_title, p.full_content
                FROM child_chunks c
                JOIN parent_documents p ON c.parent_id = p.id;
            """)
            rows = cur.fetchall()

        return [
            {
                "child_id": row[0],
                "parent_id": row[1],
                "chunk_content": row[2],
                "embedding_json": row[3],
                "source_file": row[4],
                "section_title": row[5],
                "parent_content": row[6],
            }
            for row in rows
        ]
