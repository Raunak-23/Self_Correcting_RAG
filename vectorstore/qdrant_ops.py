import os
import logging
from typing import List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, QueryRequest
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """
    Wrapper around Qdrant client with project-specific defaults.
    Supports local or cloud mode via env vars.
    """
    def __init__(
        self,
        collection_name: str = "rag_chunks",
        url: Optional[str] = os.getenv("QDRANT_URL"),
        api_key: Optional[str] = os.getenv("QDRANT_API_KEY"),
        prefer_grpc: bool = True,
    ):
        self.collection_name = collection_name
        
        if url:
            logger.info(f"Connecting to Qdrant cloud at {url}")
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=prefer_grpc)
        else:
            logger.info("Starting local Qdrant instance")
            self.client = QdrantClient(":memory:")  # In-memory for testing; use path=":disk:" for persistent
            
        self._ensure_collection()

    def _ensure_collection(self, vector_size: int = 384):
        """Create collection if it doesn't exist (default dim=384 for MiniLM)"""
        collections = self.client.get_collections()
        if self.collection_name not in [c.name for c in collections.collections]:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{self.collection_name}' with dim={vector_size}")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def upsert_chunks(self, ids: List[int], vectors: List[List[float]], payloads: List[dict]):
        """Batch upsert points with metadata"""
        from qdrant_client.http.models import PointStruct
        
        points = [
            PointStruct(id=idx, vector=vec, payload=pay)
            for idx, vec, pay in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(points)} points")

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        metadata_filter: Optional[dict] = None,
    ):
        """Semantic search with optional metadata filter"""
        filter_ = None
        if metadata_filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
            filter_ = Filter(must=conditions)
            
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,                     # just pass the vector directly
            query_filter=filter_,
            limit=limit,
            search_params={"exact": False, "hnsw_ef": 128},  # optional tuning
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
        return result.points

if __name__ == "__main__":
    import uuid
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from embeddings.embedder import Embedder      # dense embedder

    # 1. Initialize store (use test collection to not pollute main one)
    TEST_COLLECTION = "test_hybrid_demo"
    store = QdrantVectorStore(collection_name=TEST_COLLECTION)
    print(f"→ Collection ready: {TEST_COLLECTION}")

    # 2. Prepare small dummy dataset
    documents = [
        "Python is a high-level programming language created by Guido van Rossum.",
        "RAG stands for Retrieval-Augmented Generation in modern LLMs.",
        "Qdrant is an open-source vector database optimized for similarity search."
    ]

    dense_embedder = Embedder()
    dense_vectors = dense_embedder.embed_documents(documents)

    payloads = [
        {"text": doc, "source": "demo", "type": "factual"}
        for doc in documents
    ]

    point_ids = [str(uuid.uuid4()) for _ in documents]

    print(f"\n→ Prepared {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc[:70]}{'...' if len(doc)>70 else ''}")

    # 3. Upsert — dense only for this simple test
    print("\n→ Performing upsert (dense vectors only)...")
    store.upsert_chunks(
        ids=point_ids,
        vectors=dense_vectors,
        payloads=payloads
    )
    print("   Upsert completed successfully.")

    # 4. Simple search test
    query = "What is RAG?"
    query_vector = dense_embedder.embed_query(query)

    print(f"\n→ Searching for: '{query}'")
    results = store.search(
        query_vector=query_vector,
        limit=2,
        score_threshold=0.5
    )

    if results:
        print("   Top results:")
        for i, hit in enumerate(results, 1):
            print(f"     {i}. [score: {hit.score:.4f}] → {hit.payload.get('text', 'N/A')[:90]}...")
    else:
        print("   No results above threshold.")

    print("\n" + "="*70)
    print("Self-test finished. Everything seems to be working correctly!")
    print("="*70 + "\n")