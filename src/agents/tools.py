import logging

from pydantic import BaseModel
from pydantic import ValidationError
from src.retrieval.two_stage_retriever import TwoStageRetriever
from src.generation.answer_generator import AnswerGenerator
from src.generation.answer_generator import Answer

logger = logging.getLogger("RetrievalTool")

class SearchDisclosuresInput(BaseModel):
    query: str
    final_k: int = 5



class RetrievalToolError(Exception):
    pass

class RetrievalTool:
    
    def __init__(self):
        self.retriever = TwoStageRetriever()
        self.generator = AnswerGenerator()
         
    def retrieve(self, input: SearchDisclosuresInput) -> Answer:
        try:
            retrievalResult = self.retriever.retrieve(input.query, candidate_k=20, final_k = input.final_k)

            chunks =  retrievalResult["reranked"]
            answer = self.generator.generate_answer(input.query, chunks)

            return answer
        
        except ValidationError as e:
            logger.error(f"Validation failed for query={input.query!r}, {e}")
            raise e
        
        except Exception as e:
            logger.error(f"Search failed for query={input.query!r}, {e}")
            raise RetrievalToolError(f"Search failed for query={input.query!r}") from e
