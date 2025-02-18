from fastapi import HTTPException
from app.models.schemas import EmbedDocumentsRequest, SearchDocumentsRequest
from chonkie import SemanticChunker
from openai import OpenAI

import os
from pymilvus import (
    Collection,
)


class EmbeddingService:
    def __init__(self):
        self.semantic_chunker = SemanticChunker(
            embedding_model="minishlab/potion-base-8M",
            threshold=0.5,
            chunk_size=512,
            min_sentences=1,
        )
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def embed_documents(self, request: EmbedDocumentsRequest):
        try:
            chunks = self.semantic_chunker.chunk(request.documents)
            print(chunks)
            # for chunk in chunks:
            #     embeddings = self.client.embeddings.create(
            #         input=chunk,
            #         model="text-embedding-ada-002",
            #     )
            #     collection = Collection(request.collection_name)
            #     entities = [
            #         {"metadata": chunk, "vector": embedding.data[0].embedding}
            #         for chunk, embedding in zip(chunks, embeddings)
            #     ]
            # collection.insert(entities)
            
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def search_documents(self, request: SearchDocumentsRequest):
        try:
            collection = Collection(request.collection_name)
            embeded_query = self.client.embeddings.create(
                input=request.query,
                model="text-embedding-ada-002",
            )
            results = collection.search(
                data=[embeded_query.data[0].embedding],
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=request.limit,
            )
            results = [
                {
                    "metadata": result.entity.get("metadata"),
                    "score": result.score,
                }
                for result in results
            ]
            return {"results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
