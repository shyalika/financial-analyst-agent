import os
import re
import numpy as np
from openai import OpenAI
from src.ingestion.vector_store import SQLiteVectorStore
from src.ingestion.pdf_table_extraction import extract_structured_text
from src.utils.billing import TokenBudgetController

class StructuralIngestionPipeline:
    """
    Table-aware extraction (pdfplumber, src/ingestion/pdf_table_extraction.py):
    detected tables/stat-cards are converted to clean label:value chunks;
    narrative text is chunked separately via the semantic-breakpoint splitter
    below. Embedding calls are batched (see SPEC.md's consolidation section)
    rather than one API call per sentence.
    """
    def __init__(self):
        self.db = SQLiteVectorStore()
        self.finops = TokenBudgetController()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embedding_vector(self, text: str, model: str = "text-embedding-3-small") -> list:
        """Single-text embedding call. Used by query-side retrieval modules;
        ingestion's bulk paths use get_embedding_vectors (batched) instead."""
        clean_text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[clean_text], model=model)

        self.finops.update_usage_and_verify(
            model="gpt-4o-mini",
            prompt=response.usage.prompt_tokens,
            completion=0
        )
        return response.data[0].embedding

    def get_embedding_vectors(self, texts: list, model: str = "text-embedding-3-small", batch_size: int = 250) -> list:
        """Batched embedding: one OpenAI request per batch_size texts instead
        of one request per text. Returns embeddings in input order."""
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            clean_batch = [t.replace("\n", " ") for t in batch]
            response = self.client.embeddings.create(input=clean_batch, model=model)

            self.finops.update_usage_and_verify(
                model="gpt-4o-mini",
                prompt=response.usage.prompt_tokens,
                completion=0
            )
            all_embeddings.extend(item.embedding for item in response.data)

        return all_embeddings

    def _cosine_distance(self, v1, v2) -> float:
        """Calculates spatial cosine distance (1 - cosine_similarity) between two vectors."""
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 1.0
        similarity = dot_product / (norm_v1 * norm_v2)
        return float(1.0 - similarity)

    def segment_text_semantically(self, text: str, threshold: float = 0.35) -> list:
        """
        Groups sentences by vector proximity and creates splits when a topic shifts.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= 1:
            return sentences

        print(f" -> Document split into {len(sentences)} sentences. Embedding in batches to analyze semantic drift...")
        embeddings = self.get_embedding_vectors(sentences)

        chunks = []
        current_chunk = [sentences[0]]

        for i in range(len(sentences) - 1):
            distance = self._cosine_distance(embeddings[i], embeddings[i + 1])
            if distance > threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i + 1]]
                print("   ⚠️ Semantic Breakpoint Triggered! Slicing topic boundary.")
            else:
                current_chunk.append(sentences[i + 1])

        chunks.append(" ".join(current_chunk))
        return chunks

    def _extract_structured_text(self, file_path: str) -> dict:
        """Dispatches on file extension. PDFs get table-aware extraction
        (pdfplumber); plaintext files are treated as narrative-only."""
        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            return extract_structured_text(file_path)
        elif extension == ".txt":
            with open(file_path, 'r') as f:
                return {"narrative_text": f.read(), "table_blocks": []}
        else:
            raise ValueError(f"Unsupported source asset type: {extension}")

    def _ingest_narrative(self, source_file: str, narrative_text: str):
        parent_blocks = self.segment_text_semantically(narrative_text, threshold=0.35)

        # Build every parent's child windows first, so embedding happens in
        # batches instead of one API call per window.
        all_child_windows = []
        parent_child_ranges = []
        for parent_context in parent_blocks:
            sentences_in_parent = [s.strip() for s in re.split(r'(?<=[.!?])\s+', parent_context) if s.strip()]
            child_windows = [
                " ".join(sentences_in_parent[i:i + 3])
                for i in range(0, len(sentences_in_parent), 3)
            ]
            start = len(all_child_windows)
            all_child_windows.extend(child_windows)
            parent_child_ranges.append((start, len(all_child_windows)))

        print(f" -> Embedding {len(all_child_windows)} narrative child chunks in batches...")
        all_embeddings = self.get_embedding_vectors(all_child_windows)

        for idx, (parent_context, (start, end)) in enumerate(zip(parent_blocks, parent_child_ranges)):
            print(f"\nProcessing Narrative Parent Block #{idx + 1}...")
            child_tuples = list(zip(all_child_windows[start:end], all_embeddings[start:end]))

            self.db.insert_document_pipeline(
                source_file=source_file,
                section_title=f"Narrative Block #{idx + 1}",
                parent_text=parent_context,
                child_tuples=child_tuples
            )

    def _ingest_tables(self, source_file: str, table_blocks: list):
        if not table_blocks:
            return

        print(f" -> Embedding {len(table_blocks)} table blocks in batches...")
        embeddings = self.get_embedding_vectors(table_blocks)

        for idx, (table_text, vector) in enumerate(zip(table_blocks, embeddings)):
            print(f"\nProcessing Table Block #{idx + 1}...")
            self.db.insert_document_pipeline(
                source_file=source_file,
                section_title=f"Table Block #{idx + 1}",
                parent_text=table_text,
                child_tuples=[(table_text, vector)]
            )

    def run_file_ingestion(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required source asset: {file_path}")

        print(f"\nProcessing target asset: {file_path}...")

        structured = self._extract_structured_text(file_path)
        source_file = os.path.basename(file_path)

        self._ingest_narrative(source_file, structured["narrative_text"])
        self._ingest_tables(source_file, structured["table_blocks"])

        print("\n🚀 Ingestion Run Final Metrics:")
        print(f"Total Run Costs: ${self.finops.usage.total_cost_usd:.6f} USD")
