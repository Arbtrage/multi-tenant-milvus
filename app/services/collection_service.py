from fastapi import HTTPException
from pymilvus import Collection, utility
from app.models.schemas import CreateCollectionRequest
from app.core.config import COLLECTIONS_CONFIG
from app.core.milvus_client import get_collection
from pymilvus import (
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
)


class CollectionService:
    @staticmethod
    async def list_collections():
        try:
            collections = utility.list_collections()
            collections_info = []
            for name in collections:
                collection = Collection(name)
                schema = collection.schema
                collections_info.append(
                    {
                        "name": name,
                        "description": schema.description,
                        "dimension": schema.fields[1].dim,
                    }
                )
            return {"collections": collections_info}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def get_collection(collection_name: str):
        try:
            collection = get_collection(collection_name)
            return {"collection": collection}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def create_collection(request: CreateCollectionRequest):
        try:
            if request.name in COLLECTIONS_CONFIG:
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection {request.name} already exists in configuration",
                )

            if utility.has_collection(request.name):
                raise HTTPException(
                    status_code=400,
                    detail=f"Collection {request.name} already exists in Milvus",
                )

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=request.dimension
                ),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields=fields, description=request.description)
            collection = Collection(name=request.name, schema=schema)
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024},
            }
            collection.create_index(field_name="vector", index_params=index_params)

            COLLECTIONS_CONFIG[request.name] = {
                "dim": request.dimension,
                "description": request.description,
            }

            return {
                "status": "success",
                "message": f"Collection {request.name} created successfully",
                "details": {
                    "name": request.name,
                    "dimension": request.dimension,
                    "description": request.description,
                },
            }

        except Exception as e:
            print(e)
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=str(e))
