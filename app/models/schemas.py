from typing import List, Dict, Optional
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 10
    metadata_filter: Optional[Dict] = None
    collection_name: str = "products"


class SearchResponse(BaseModel):
    ids: List[int]
    distances: List[float]
    metadata_ids: List[str]


class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int = 1536
    description: str = "Vector embeddings collection"


class EmbedDocumentsRequest(BaseModel):
    documents: str = "This is a test document"
    collection_name: str = "products"


class SearchDocumentsRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 10
    metadata_filter: Optional[Dict] = None
    collection_name: str = "products"


