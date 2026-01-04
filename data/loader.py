import os
import glob
import hashlib
import tiktoken
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Chunking Profiles for Adaptive Retrieval
CHUNK_PROFILES = {
    "factual": {"chunk_size": 256, "chunk_overlap": 30},
    "conceptual": {"chunk_size": 512, "chunk_overlap": 50},
    "long_context": {"chunk_size": 1024, "chunk_overlap": 100}
}

TOKEN_ENCODING = "cl100k_base"  # Standard for OpenAI/modern models

def get_token_count(text: str) -> int:
    """Returns accurate token count for a string."""
    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    return len(encoding.encode(text))

def generate_doc_id(content: str) -> str:
    """Generates a stable hash ID based on document content."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_single_document(file_path: str) -> List[Document]:
    """
    Loads a single document based on its file extension.
    Supports: .pdf, .docx, .txt, .md
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt' or ext == '.md':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            logger.warning(f"Skipping unsupported file extension: {ext} for {file_path}")
            return []
            
        docs = loader.load()
        # Initial metadata application
        for doc in docs:
            doc.metadata['source'] = file_path
            doc.metadata['filename'] = os.path.basename(file_path)
            doc.metadata['extension'] = ext
        return docs
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {str(e)}")
        return []

def load_and_chunk_documents(raw_docs_dir: str) -> List[Document]:
    """
    Traverses the directory, loads documents, and creates multiple chunk variations 
    (factual, conceptual, long_context) for each document.
    """
    all_chunks = []
    
    if not os.path.exists(raw_docs_dir):
        logger.error(f"Directory path does not exist: {raw_docs_dir}")
        return []

    # 1. Load all raw documents first
    raw_documents = []
    for root, _, files in os.walk(raw_docs_dir):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
                file_path = os.path.join(root, file)
                logger.info(f"Processing: {file_path}")
                loaded_docs = load_single_document(file_path)
                raw_documents.extend(loaded_docs)
    
    if not raw_documents:
        logger.warning("No valid documents found to process.")
        return []

    logger.info(f"Total raw documents loaded: {len(raw_documents)}")

    # 2. Apply Adaptive Chunking
    for original_doc in raw_documents:
        doc_id = generate_doc_id(original_doc.page_content)
        
        for profile_name, settings in CHUNK_PROFILES.items():
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings["chunk_size"],
                chunk_overlap=settings["chunk_overlap"],
                separators=["\n\n", "\n", " ", ""],
                length_function=len
            )
            
            splits = splitter.split_text(original_doc.page_content)
            
            for i, split_content in enumerate(splits):
                chunk_metadata = original_doc.metadata.copy()
                chunk_metadata.update({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "chunk_type": profile_name,
                    "tokens": get_token_count(split_content),
                    "profile_chunk_size": settings["chunk_size"]
                })
                
                chunk_doc = Document(
                    page_content=split_content,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk_doc)
                
    logger.info(f"Generated {len(all_chunks)} total chunks across {len(CHUNK_PROFILES)} profiles.")
    
    return all_chunks

if __name__ == "__main__":
    # Test execution
    import sys
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw_docs"
    
    print(f"Testing adaptive loader on: {test_dir}")
    chunks = load_and_chunk_documents(test_dir)
    
    if chunks:
        print(f"\nExample Metadata (Factual):")
        factual = next((c for c in chunks if c.metadata['chunk_type'] == 'factual'), None)
        if factual:
            print(factual.metadata)
            
        print(f"\nExample Metadata (Long Context):")
        long_ctx = next((c for c in chunks if c.metadata['chunk_type'] == 'long_context'), None)
        if long_ctx:
            print(long_ctx.metadata)