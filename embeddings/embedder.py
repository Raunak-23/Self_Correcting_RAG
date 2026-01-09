import os
import logging
from typing import List
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

load_dotenv()  # Load any env vars if needed (e.g., for HF token if private models)

# Logging setup (consistent with loader.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default embedding model (lightweight, 384-dim, semantic-focused)
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class Embedder:
    """
    Handles text embedding using HuggingFace models via LangChain integration.
    Supports batch embedding for documents and single-query embedding.
    Automatically detects device (GPU if available) for efficiency.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, batch_size: int = 64, normalize: bool = True):
        """
        Initializes the embedder.
        
        Args:
            model_name (str): HuggingFace model name (default: all-MiniLM-L6-v2).
            batch_size (int): Max batch size for embedding to avoid OOM.
            normalize (bool): Normalize embeddings (L2 norm) for cosine similarity.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device} for embeddings")
        
        # Load via LangChain for compatibility with vector stores
        self.embedder = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": self.normalize}
        )
        
        # For manual control if needed (e.g., custom tokenization)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts in batches.
        
        Args:
            texts (List[str]): List of document/chunk contents to embed.
        
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        if not texts:
            return []
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self.embedder.embed_documents(batch)
                embeddings.extend(batch_embeddings)
                logger.info(f"Embedded batch {i//self.batch_size + 1}/{len(texts)//self.batch_size + 1}")
            except Exception as e:
                logger.error(f"Error embedding batch: {str(e)}")
                # Fallback: Embed individually
                for text in batch:
                    try:
                        embeddings.append(self.embedder.embed_query(text))
                    except Exception as sub_e:
                        logger.error(f"Failed to embed text: {text[:50]}... Error: {str(sub_e)}")
                        embeddings.append([0.0] * 384)  # Zero vector as placeholder (adjust dim if model changes)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string.
        
        Args:
            query (str): The query text to embed.
        
        Returns:
            List[float]: Embedding vector.
        """
        try:
            return self.embedder.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query '{query[:50]}...': {str(e)}")
            return [0.0] * 384  # Placeholder (match model dim)

if __name__ == "__main__":
    # Quick test script
    embedder = Embedder()
    test_texts = ["This is a sample factual sentence.", "Conceptual explanation of AI."]
    embeddings = embedder.embed_documents(test_texts)
    print(f"Embedded {len(test_texts)} texts. First embedding length: {len(embeddings[0])}")
    query_emb = embedder.embed_query("What is RAG?")
    print(f"Query embedding length: {len(query_emb)}")