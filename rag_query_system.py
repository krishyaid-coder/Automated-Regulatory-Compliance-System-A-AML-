"""
RAG Query System for Regulatory Compliance
Simple functions for querying documents with Gemini
"""

import json
import pickle
import os
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Configuration
VECTOR_STORE_DIR = 'vector_store'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_LLM_MODEL = 'gemini-2.5-flash'
TOP_K_RETRIEVAL = 5

# Global variables (loaded once)
_index = None
_chunk_metadata = None
_embedding_model = None
_llm_model = None
_genai_model = None


def load_vector_store(vector_store_dir=VECTOR_STORE_DIR):
    """Load FAISS index and chunk metadata."""
    global _index, _chunk_metadata
    
    index_path = os.path.join(vector_store_dir, 'faiss_index.bin')
    metadata_path = os.path.join(vector_store_dir, 'chunk_metadata.pkl')
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError("Vector store not found. Run rag_pipeline.py first.")
    
    _index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        _chunk_metadata = pickle.load(f)
    
    print(f"✓ Loaded {_index.ntotal} vectors and {len(_chunk_metadata)} chunks")


def load_embedding_model(model_name=EMBEDDING_MODEL):
    """Load embedding model."""
    global _embedding_model
    _embedding_model = SentenceTransformer(model_name)
    print(f"✓ Loaded embedding model: {model_name}")


def load_llm(model_name=DEFAULT_LLM_MODEL, api_key=None):
    """Configure Gemini client using the official SDK."""
    global _llm_model, _genai_model
    
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("⚠️  No API key. Set GOOGLE_API_KEY or pass api_key parameter")
        _llm_model = None
        _genai_model = None
        return
    
    genai.configure(api_key=api_key)
    _llm_model = model_name
    try:
        _genai_model = genai.GenerativeModel(model_name)
        print(f"✓ Gemini configured for model: {model_name}")
    except Exception as exc:
        _genai_model = None
        print(f"⚠️  Failed to initialize Gemini model '{model_name}': {exc}")


def initialize_system(vector_store_dir=VECTOR_STORE_DIR, 
                     embedding_model=EMBEDDING_MODEL,
                     llm_model=DEFAULT_LLM_MODEL,
                     api_key=None):
    """Initialize all components."""
    load_vector_store(vector_store_dir)
    load_embedding_model(embedding_model)
    load_llm(llm_model, api_key)


def retrieve_chunks(query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
    """Retrieve relevant chunks for a query."""
    if _index is None or _embedding_model is None:
        raise ValueError("System not initialized. Call initialize_system() first.")
    
    # Encode query
    query_embedding = _embedding_model.encode([query])[0].astype('float32')
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search
    distances, indices = _index.search(query_embedding, top_k)
    
    # Get chunks with metadata
    retrieved_chunks = []
    for distance, idx in zip(distances[0], indices[0]):
        chunk = _chunk_metadata[idx].copy()
        chunk['similarity_score'] = float(1 / (1 + distance))
        chunk['distance'] = float(distance)
        retrieved_chunks.append(chunk)
    
    return retrieved_chunks


def extract_citation(chunk: Dict[str, Any]) -> str:
    """Extract citation information from chunk."""
    filename = chunk.get('filename', 'Unknown')
    chunk_id = chunk.get('chunk_id', 'Unknown')
    metadata = chunk.get('metadata', {})
    
    sections = metadata.get('sections', [])
    if sections:
        unique_sections = list(set(sections))
        section_str = ", ".join(unique_sections[:3])
        return f"{filename}, {section_str}"
    
    return f"{filename}, Chunk {chunk_id}"


def generate_answer(question: str, context: str, citations: List[Dict]) -> str:
    """Generate answer using Gemini."""
    if _genai_model is None:
        return "LLM not available. Please initialize with load_llm(). Showing top chunk instead:\n\n" + context.split("\n\n")[0]
    
    prompt = f"""You are a compliance expert assistant. Answer the question based ONLY on the provided regulatory context. 

CRITICAL RULES:
1. ONLY use information from the provided context. Do not use any external knowledge.
2. If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
3. For every claim or fact, cite the exact source using the citation format: [Citation X] where X is the citation number.
4. Be precise and factual. Do not make assumptions or add information not in the context.

Context:
{context}

Citations:
{chr(10).join([f"[Citation {c['citation_id']}] {c['source']}" for c in citations])}

Question: {question}

Answer (with citations):"""

    try:
        response = _genai_model.generate_content(prompt)
        candidates = getattr(response, "candidates", [])
        if candidates and candidates[0].content.parts:
            return candidates[0].content.parts[0].text
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        fallback = retrieved_chunks_text_preview(citations)
        return f"Error generating answer: {e}\n\nTop context snippet:\n{fallback}"


def retrieved_chunks_text_preview(citations: List[Dict]) -> str:
    """Return preview text from the first citation."""
    if not citations:
        return "No context available."
    return citations[0].get("text_preview", "No preview available.")


def query(question: str, top_k: int = TOP_K_RETRIEVAL, use_llm: bool = True) -> Dict[str, Any]:
    """Query the RAG system."""
    # Retrieve chunks
    retrieved_chunks = retrieve_chunks(question, top_k)
    
    if not retrieved_chunks:
        return {
            'answer': 'No relevant information found.',
            'citations': [],
            'retrieved_chunks': [],
            'confidence': 0.0
        }
    
    # Build context and citations
    context_parts = []
    citations = []
    
    for i, chunk in enumerate(retrieved_chunks, 1):
        chunk_text = chunk.get('text', '')
        citation = extract_citation(chunk)
        context_parts.append(f"[Context {i}] {chunk_text}")
        citations.append({
            'citation_id': i,
            'source': citation,
            'filename': chunk.get('filename', 'Unknown'),
            'similarity_score': chunk.get('similarity_score', 0.0),
            'text_preview': chunk_text[:200] + '...' if len(chunk_text) > 200 else chunk_text
        })
    
    context = "\n\n".join(context_parts)
    
    # Generate answer
    if use_llm and _llm_model:
        answer = generate_answer(question, context, citations)
    else:
        answer = f"Based on the retrieved information:\n\n{retrieved_chunks[0].get('text', '')[:500]}..."
    
    # Calculate confidence
    avg_similarity = sum(c['similarity_score'] for c in citations) / len(citations) if citations else 0.0
    
    return {
        'answer': answer,
        'citations': citations,
        'retrieved_chunks': retrieved_chunks,
        'confidence': avg_similarity,
        'num_chunks_retrieved': len(retrieved_chunks)
    }


def main():
    """Example usage."""
    print("="*80)
    print("RAG Query System")
    print("="*80)
    
    # Initialize
    try:
        initialize_system()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Test queries
    test_queries = [
        "What are the customer due diligence requirements?",
        "What is required for KYC compliance?",
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = query(q, use_llm=False)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Citations: {len(result['citations'])}")


if __name__ == "__main__":
    main()
