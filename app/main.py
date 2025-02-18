from fastapi import FastAPI
from pymilvus import connections
from contextlib import asynccontextmanager

from app.core.config import MILVUS_HOST, MILVUS_PORT
from app.api.routes import collection_routes, embedding_routes
from app.core.milvus_client import start_background_unloader


@asynccontextmanager
async def lifespan(app: FastAPI):
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    start_background_unloader()
    yield
    connections.disconnect("default")


app = FastAPI(title="Vector Search API", version="1.0.0", lifespan=lifespan)

app.include_router(collection_routes.router)
# app.include_router(embedding_routes.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
