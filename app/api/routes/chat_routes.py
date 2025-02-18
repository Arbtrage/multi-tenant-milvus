from fastapi import APIRouter
from app.models.schemas import EmbedDocumentsRequest
from app.services.embedding_service import EmbeddingService

router = APIRouter(prefix="/chat", tags=["Chat"])


# @router.post("/")
# async def chat(request: ChatRequest):
#     """Chat"""
#     return await ChatService.chat(request)