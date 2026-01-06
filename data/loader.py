import os
import time
import glob
import hashlib
import tiktoken
from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import logging
from dotenv import load_dotenv
from transformers import pipeline
import json
from pathlib import Path

load_dotenv()  # Load API keys from .env

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

CACHE_DIR = "data/processed_chunks/classification_cache"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

def get_token_count(text: str) -> int:
    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    return len(encoding.encode(text))

def generate_doc_id(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def classify_doc_type(content: str, filename: str, use_huggingface: bool = False) -> str:
    """
    Classifies document type using Groq LLM (default) or HuggingFace zero-shot classifier (fallback).
    Uses filename-based caching for readability.
    """
    # Sanitize filename for safe path (replace invalid chars)
    safe_filename = "".join(c if c.isalnum() or c in ['.', '_'] else '_' for c in filename)
    cache_path = f"{CACHE_DIR}/{safe_filename}.json"
    
    # Check for existing cache
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)['type']
    
    total_len = len(content)
    samples = [
        content[:500],                                      # Intro/abstract
        content[total_len//2 - 250: total_len//2 + 250],     # Middle section
        content[-500:]                                      # Conclusion/references
    ]
    
    votes = []
    for excerpt in samples:
        if len(excerpt.strip()) < 50:  # Skip empty samples
            continue
        if not use_huggingface:
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
            prompt = ChatPromptTemplate.from_template(
                "Classify this document excerpt strictly as one type: "
                "'factual' (lists facts, data, definitions, timelines), "
                "'conceptual' (explains ideas, comparisons, theories, processes), or "
                "'long_context' (narratives, stories, detailed overviews, long descriptions). "
                "Output ONLY the type, nothing else.\n\nExcerpt: {excerpt}"
            )
            chain = prompt | llm
            response = chain.invoke({"excerpt": excerpt}).content.strip()
        else:
            classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-small-zeroshot-v1")
            labels = ["factual: lists facts, data, definitions, timelines",
                    "conceptual: explains ideas, comparisons, theories, processes",
                    "long_context: narratives, stories, detailed overviews, long descriptions"]
            result = classifier(excerpt, candidate_labels=labels, multi_label=False)
            response = result['labels'][0].split(':')[0].strip()
        if response in CHUNK_PROFILES:
            votes.append(response)
        
    # Majority vote with fallback
    if votes:
        from collections import Counter
        response = Counter(votes).most_common(1)[0][0]
    else:
        response = "conceptual"
    
    # Save to cache
    with open(cache_path, 'w') as f:
        json.dump({'type': response}, f)
    
    return response
    
# In load_and_chunk_documents, call as: doc_type = classify_doc_type(doc_content)  
# Or pass use_huggingface=True for local

def load_single_document(file_path: str) -> List[Document]:
    """
    Loads a single document and merges all pages into ONE Document object.
    This enables whole-document classification while preserving source metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)  # Default mode="page" gives per-page
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt' or ext == '.md':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            logger.warning(f"Skipping unsupported file extension: {ext} for {file_path}")
            return []
        
        # Load per-page docs first
        page_docs = loader.load()
        
        if not page_docs:
            return []
        
        # Merge all pages into single document content
        full_content = "\n\n".join(doc.page_content for doc in page_docs)
        
        # Use metadata from first page (or enrich as needed)
        base_metadata = page_docs[0].metadata.copy()
        base_metadata.update({
            'source': file_path,
            'filename': os.path.basename(file_path),
            'extension': ext,
            'total_pages': len(page_docs)  # Bonus: track page count
        })
        
        merged_doc = Document(page_content=full_content, metadata=base_metadata)
        return [merged_doc]  # Return list with single merged Document
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {str(e)}")
        return []

def load_and_chunk_documents(raw_docs_dir: str, dynamic_upload: bool = False) -> List[Document]:
    """
    Loads docs, classifies type, applies targeted chunk profiles, adds metadata.
    If dynamic_upload=True (e.g., from Streamlit), process only new files.
    """
    all_chunks = []
    
    if not os.path.exists(raw_docs_dir):
        logger.error(f"Directory path does not exist: {raw_docs_dir}")
        return []

    # Load raw documents (filter for new if dynamic)
    raw_documents = []
    for root, _, files in os.walk(raw_docs_dir):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx', '.txt', '.md')):
                file_path = os.path.join(root, file)
                if dynamic_upload and os.path.exists(f"processed_chunks/{file}.json"):  # Skip processed
                    continue
                logger.info(f"Processing: {file_path}")
                loaded_docs = load_single_document(file_path)
                raw_documents.extend(loaded_docs)
    
    if not raw_documents:
        return []

    # Apply Adaptive Chunking with Filtering
    for original_doc in raw_documents:
        doc_content = original_doc.page_content
        doc_id = generate_doc_id(doc_content)
        time.sleep(2)  # Delay to avoid rate limits
        doc_type = classify_doc_type(doc_content, original_doc.metadata['filename'])  # Auto-filter type
        logger.info(f"Classified {original_doc.metadata['filename']} as {doc_type}")
        
        # Apply only relevant profiles (primary + one fallback)
        profiles_to_apply = [doc_type, "conceptual"] if doc_type != "conceptual" else [doc_type]
        
        for profile_name in profiles_to_apply:
            settings = CHUNK_PROFILES[profile_name]
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings["chunk_size"],
                chunk_overlap=settings["chunk_overlap"],
                separators=["\n\n", "\n", " ", ""],
                length_function=get_token_count  # Token-based for accuracy
            )
            
            splits = splitter.split_text(doc_content)
            
            for i, split_content in enumerate(splits):
                chunk_metadata = original_doc.metadata.copy()
                chunk_metadata.update({
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "chunk_type": profile_name,
                    "tokens": get_token_count(split_content),
                    "profile_chunk_size": settings["chunk_size"],
                    "doc_primary_type": doc_type  # For advanced filtering
                })
                
                chunk_doc = Document(page_content=split_content, metadata=chunk_metadata)
                all_chunks.append(chunk_doc)
                
    logger.info(f"Generated {len(all_chunks)} chunks.")
    return all_chunks

if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="Test the adaptive document loader.")
    parser.add_argument(
        "raw_docs_dir",
        nargs="?",
        default="data/raw_docs",  # Adjust if your folder is named differently (e.g., raw_data)
        help="Path to the raw documents directory (default: data/raw_docs)"
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Force use HuggingFace zero-shot classifier instead of Groq (for offline/no-API testing)"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Simulate dynamic upload mode (skip already processed files)"
    )
    args = parser.parse_args()

    print(f"Testing loader on directory: {args.raw_docs_dir}")
    print(f"Using {'HuggingFace' if args.use_hf else 'Groq LLM'} for classification")
    print(f"Dynamic mode: {args.dynamic}\n")

    # Patch classify_doc_type temporarily if forcing HF
    if args.use_hf:
        original_classify = classify_doc_type
        def classify_doc_type(content: str, use_huggingface: bool = False) -> str:
            return original_classify(content, use_huggingface=True)

    chunks = load_and_chunk_documents(args.raw_docs_dir, dynamic_upload=args.dynamic)

    if not chunks:
        print("No chunks generated — check directory path and file formats!")
        exit()

    print(f"\nSuccessfully generated {len(chunks)} chunks from {len(set(c.metadata['filename'] for c in chunks))} documents.\n")

    # Summary statistics
    doc_types = Counter(c.metadata['doc_primary_type'] for c in chunks)
    chunk_types = Counter(c.metadata['chunk_type'] for c in chunks)
    token_stats = [c.metadata['tokens'] for c in chunks]

    print("Document Primary Type Distribution:")
    for typ, count in doc_types.items():
        print(f"  - {typ}: {count // len(CHUNK_PROFILES)} documents")  # Approx, since multiple profiles

    print("\nChunk Type Distribution:")
    for typ, count in chunk_types.items():
        print(f"  - {typ}: {count} chunks")

    print(f"\nToken Statistics:")
    print(f"  - Average tokens per chunk: {sum(token_stats)/len(token_stats):.1f}")
    print(f"  - Min/Max tokens: {min(token_stats)} / {max(token_stats)}")

    # Show a few example chunks
    print("\nExample Chunks:")
    seen_files = set()
    for chunk in chunks[:10]:  # Show up to 10 diverse examples
        filename = chunk.metadata['filename']
        if filename in seen_files:
            continue
        seen_files.add(filename)
        print(f"\n--- {filename} ---")
        print(f"Primary Type: {chunk.metadata['doc_primary_type']}")
        print(f"Chunk Type: {chunk.metadata['chunk_type']} (size {chunk.metadata['profile_chunk_size']})")
        print(f"Tokens: {chunk.metadata['tokens']}")
        print(f"Preview: {chunk.page_content[:300]}...\n")