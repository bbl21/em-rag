"""Protocol-independent domain objects for the EM RAG product."""

from .errors import ProductError
from .models import AnswerResponse, Evidence, RetrievalRequest, RetrievalResponse

__all__ = ["AnswerResponse", "Evidence", "ProductError", "RetrievalRequest", "RetrievalResponse"]
