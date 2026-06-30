"""
Ingest script — loads Blendata Enterprise v4.6.0 documentation into Qdrant

Usage:
    python scripts/ingest.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import settings
from app.dependencies import get_embeddings

DATA_DIR = Path(__file__).parent.parent / "data"
SOURCE_FILE = DATA_DIR / "bde460_content.md"

# Top-level section headings used to tag chunks with a category
_SECTION_PATTERNS = [
    (r"getting.started",          "Getting Started"),
    (r"import.data|import.dataset","Import Data"),
    (r"explore.*process|sql.editor|notebook|data.exploration|data.preparation|view.table|aggregate.table|export", "Explore & Process"),
    (r"visualization|dashboard",  "Visualization & Dashboard"),
    (r"workflow",                 "Workflow Management"),
    (r"data.catalog|data.lineage|create.table", "Data Catalog"),
    (r"data.policy|service.management|schedule.job|chain.job", "Data Policy & Services"),
    (r"admin|user.management|realm|ldap|license|storage|schema|sso|access.token", "Administration"),
    (r"integration|jdbc|odbc|tableau|power.bi|dbeaver", "Integration"),
    (r"general.references|architecture|api.document|security|logging|tuning|cli", "General References"),
    (r"setting|configuration|kafka.management|module", "Settings & Config"),
]


def _guess_category(text: str) -> str:
    lower = text.lower()
    for pattern, label in _SECTION_PATTERNS:
        if re.search(pattern, lower):
            return label
    return "General"


def load_bde460() -> list[Document]:
    text = SOURCE_FILE.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
    )

    raw_chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": "bde460_content.md"}],
    )

    # Skip the table-of-contents block (all link lines, no real content)
    docs = []
    for chunk in raw_chunks:
        lines = [l for l in chunk.page_content.splitlines() if l.strip()]
        non_link_lines = [l for l in lines if not re.match(r"^-\s*\[", l)]
        if len(non_link_lines) < 3:
            continue  # skip TOC-only chunks

        category = _guess_category(chunk.page_content)
        chunk.metadata["category"] = category
        docs.append(chunk)

    print(f"Loaded {len(docs)} chunks from {SOURCE_FILE.name}")
    return docs


def create_collection(client: QdrantClient, embedding_dim: int) -> None:
    existing = [c.name for c in client.get_collections().collections]

    if settings.collection_name in existing:
        print(f"Collection '{settings.collection_name}' exists — recreating...")
        client.delete_collection(settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )
    print(f"Created collection: {settings.collection_name}")


def main():
    print("Starting data ingestion — Blendata Enterprise v4.6.0 docs")
    print(f"  Qdrant:     {settings.qdrant_url}")
    print(f"  Collection: {settings.collection_name}")
    print(f"  Model:      {settings.embedding_model}\n")

    docs = load_bde460()
    print(f"Total chunks: {len(docs)}")

    print("\nLoading embedding model (first run downloads ~90MB)...")
    embeddings = get_embeddings()
    embedding_dim = len(embeddings.embed_query("test"))
    print(f"Embedding dimension: {embedding_dim}")

    client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    create_collection(client, embedding_dim)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=settings.collection_name,
        embedding=embeddings,
    )

    print(f"\nIngesting {len(docs)} chunks...")
    batch_size = 50
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  {min(i + batch_size, len(docs))}/{len(docs)} inserted")

    total = client.count(settings.collection_name).count
    print(f"\nDone! {total} vectors stored in Qdrant.")


if __name__ == "__main__":
    main()
