from fastapi import APIRouter
from app.models.schemas import CreateCollectionRequest
from app.services.collection_service import CollectionService

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.get("/")
async def list_collections():
    """List all available collections from Milvus"""
    return await CollectionService.list_collections()


@router.post("/")
async def create_collection(request: CreateCollectionRequest):
    """Create a new collection"""
    return await CollectionService.create_collection(request)


@router.get("/{collection_name}")
async def get_collection(collection_name: str):
    """Get a collection"""
    return await CollectionService.get_collection(collection_name)

