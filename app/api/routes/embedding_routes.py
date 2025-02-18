from fastapi import APIRouter
from app.models.schemas import EmbedDocumentsRequest, SearchDocumentsRequest
from app.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])

service = EmbeddingService()

@router.post("/")
async def embed_documents(request: EmbedDocumentsRequest):
    """Embed documents"""
    
    return await service.embed_documents(request)


@router.post("/search")
async def search_documents(request: SearchDocumentsRequest):
    """Search documents"""
    return await service.search_documents(request)
