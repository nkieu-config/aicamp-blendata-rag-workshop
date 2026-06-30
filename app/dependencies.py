from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

_embeddings: HuggingFaceEmbeddings | None = None
_vectorstore: QdrantVectorStore | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> QdrantVectorStore:
    global _vectorstore
    if _vectorstore is None:
        client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
        _vectorstore = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=get_embeddings(),
        )
    return _vectorstore
