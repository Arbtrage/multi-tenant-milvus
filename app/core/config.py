from datetime import timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Milvus connection configuration
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# Collection timeout configuration
COLLECTION_TIMEOUT = timedelta(minutes=1)

# Collections configuration
COLLECTIONS_CONFIG = {
    "products": {"dim": 1536, "description": "Product embeddings"},
    "articles": {"dim": 1536, "description": "Article embeddings"},
    "images": {"dim": 1536, "description": "Image embeddings"},
    "documents": {"dim": 1536, "description": "Document embeddings"},
    "users": {"dim": 1536, "description": "User embeddings"},
}
