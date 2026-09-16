import os
import time
import cohere
from openai import OpenAI
from src.retrieval.similarity_search import SimilaritySearch
from src.retrieval.embedding_utils import embed_text
from src.utils.billing import TokenBudgetController

RERANK_MODEL = "rerank-english-v3.0"

class TwoStageRetriever:
    """
    Stage 1: wide-net vector similarity search (cheap, approximate) to get
    candidate_k candidates.
    Stage 2: Cohere Rerank (expensive, precise cross-encoder) re-scores just
    those candidates and cuts the window down to final_k.
    """
    def __init__(self, search: SimilaritySearch = None):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.finops = TokenBudgetController()
        self.search = search or SimilaritySearch()
        self.cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

    def fetch_candidates(self, query: str, candidate_k: int = 20) -> list:
        query_embedding = embed_text(self.client, self.finops, query)
        return self.search.search(query_embedding, top_k=candidate_k)

    def rerank(self, query: str, candidates: list, final_k: int = 5) -> list:
        response = self.cohere_client.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=[c["chunk_content"] for c in candidates],
            top_n=final_k,
        )

        reranked = []
        for result in response.results:
            candidate = candidates[result.index]
            reranked.append({
                **candidate,
                "relevance_score": result.relevance_score,
                "stage1_rank": result.index + 1,
            })
        return reranked

    def retrieve(self, query: str, candidate_k: int = 20, final_k: int = 5) -> dict:
        stage1_start = time.perf_counter()
        candidates = self.fetch_candidates(query, candidate_k=candidate_k)
        stage1_latency_ms = (time.perf_counter() - stage1_start) * 1000

        stage2_start = time.perf_counter()
        reranked = self.rerank(query, candidates, final_k=final_k)
        stage2_latency_ms = (time.perf_counter() - stage2_start) * 1000

        return {
            "stage1_candidates": candidates,
            "reranked": reranked,
            "stage1_latency_ms": stage1_latency_ms,
            "stage2_latency_ms": stage2_latency_ms,
        }
