import json
import os
import pickle
from typing import List, Dict, Any
from text_chunker import (
    chunk_multiple_documents,
    preview_chunks
)

# Embedding and vector store imports
from sentence_transformers import SentenceTransformer
import faiss
from faiss import IndexFlatL2
import numpy as np

# Steps to be performed:
# 1. Read the extracted text from the JSON file
# 2. Chunk the text into smaller chunks
# 3. Create embeddings for chunks
# 4. Store embeddings in vector store

# Configuration
CHUNK_SIZE = 500  # Optimal for most LLMs
CHUNK_OVERLAP = 50  # Enough context preservation
EXTRACTED_TEXT_FILE = 'extracted_text.json'
CHUNKS_OUTPUT_DIR = 'chunks'
VECTOR_STORE_DIR = 'vector_store'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Fast and efficient embedding model

# Create output directories if they don't exist
os.makedirs(CHUNKS_OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Step 1: Read the extracted text from the JSON file
print("Step 1: Reading extracted text from JSON file...")
with open(EXTRACTED_TEXT_FILE, 'r', encoding='utf-8') as f:
    extracted_data = json.load(f)

print(f"Loaded {len(extracted_data)} document(s) from {EXTRACTED_TEXT_FILE}")

# Step 2: Chunk the text into smaller chunks
print(f"\nStep 2: Chunking text (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
results = chunk_multiple_documents(
    extracted_data,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Save chunked results to JSON
chunks_output_file = os.path.join(CHUNKS_OUTPUT_DIR, 'chunked_documents.json')
with open(chunks_output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Chunked documents saved to {chunks_output_file}")

# Display summary and quality analysis
print("\n" + "="*80)
print("Chunking Summary & Quality Analysis")
print("="*80)

def analyze_chunking_quality(results):
    """Analyze the quality of chunked documents."""
    for doc_name, doc_data in results.items():
        print(f"\n{'='*80}")
        print(f"Document: {doc_name}")
        print(f"{'='*80}")
        
        chunks = doc_data['chunks']
        total_chunks = len(chunks)
        total_chars = doc_data['total_chars']
        
        # Calculate statistics
        sizes = [chunk.get('size', len(chunk.get('text', ''))) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        min_size = min(sizes) if sizes else 0
        max_size = max(sizes) if sizes else 0
        median_size = sorted(sizes)[len(sizes)//2] if sizes else 0
        
        print(f"\nStatistics:")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average chunk size: {avg_size:.0f} chars")
        print(f"   Median chunk size: {median_size} chars")
        print(f"   Min chunk size: {min_size} chars")
        print(f"   Max chunk size: {max_size} chars")
        
        # Size distribution
        size_ranges = {
            'Very Small (< 100)': sum(1 for s in sizes if s < 100),
            'Small (100-300)': sum(1 for s in sizes if 100 <= s < 300),
            'Good (300-700)': sum(1 for s in sizes if 300 <= s < 700),
            'Large (700-1000)': sum(1 for s in sizes if 700 <= s < 1000),
            'Very Large (> 1000)': sum(1 for s in sizes if s >= 1000)
        }
        
        print(f"\nize Distribution:")
        for range_name, count in size_ranges.items():
            percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
            bar = '█' * int(percentage / 2)
            print(f"   {range_name:20s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Identify issues
        # Very small chunks
        very_small = [(i, s, chunks[i].get('text', '')[:80]) 
                      for i, s in enumerate(sizes) if s < 100]
        if very_small:
            print(f"\n Very Small Chunks (< 100 chars): {len(very_small)}")
            for idx, size, preview in very_small[:3]:
                print(f"   Chunk {idx}: {size} chars - '{preview}...'")
        
        # Very large chunks
        very_large = [(i, s, chunks[i].get('text', '')[:80]) 
                      for i, s in enumerate(sizes) if s > 800]
        if very_large:
            print(f"\n Very Large Chunks (> 800 chars): {len(very_large)}")
            for idx, size, preview in very_large[:3]:
                print(f"   Chunk {idx}: {size} chars - '{preview}...'")
        
        # Check chunk boundaries (mid-sentence breaks)
        boundary_issues = []
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].get('text', '').strip()
            chunk2_text = chunks[i+1].get('text', '').strip()
            
            if chunk1_text and chunk2_text:
                last_char = chunk1_text[-1] if chunk1_text else ''
                first_char = chunk2_text[0] if chunk2_text else ''
                
                if last_char not in '.!?;:\n' and first_char.isupper() and last_char.isalnum():
                    boundary_issues.append(i)
        
        if boundary_issues:
            print(f"\n Mid-Sentence Breaks: {len(boundary_issues)} detected")
            for idx in boundary_issues[:3]:
                chunk1 = chunks[idx].get('text', '')[-50:]
                chunk2 = chunks[idx+1].get('text', '')[:50]
                print(f"   Between chunks {idx}-{idx+1}: ...'{chunk1}' | '{chunk2}'...")
        else:
            print(f"\n✓ Chunk boundaries look good")
        
        # Check for table of contents
        toc_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            if ('CHAPTER' in text or 'CONTENTS' in text) and ('...' in text or text.count('.') > 5):
                toc_chunks.append(i)
        
        if toc_chunks:
            print(f"\n  Table of Contents Chunks: {len(toc_chunks)}")
            for idx in toc_chunks[:2]:
                print(f"   Chunk {idx}: '{chunks[idx].get('text', '')[:100]}...'")
        
        # Overall assessment
        print(f"\n{'='*80}")
        print("Overall Assessment:")
        
        score = 100
        if len(very_small) > total_chunks * 0.05:  # More than 5% very small
            score -= 20
            print("  Too many very small chunks")
        if len(very_large) > total_chunks * 0.05:  # More than 5% very large
            score -= 20
            print("   Too many very large chunks")
        if avg_size < 300:
            score -= 15
            print("   Average chunk size too small")
        if avg_size > 700:
            score -= 15
            print("   Average chunk size too large")
        if boundary_issues:
            score -= 10
            print("   Mid-sentence breaks detected")
        
        if score >= 80:
            print(f"   Quality Score: {score}/100 - GOOD")
        elif score >= 60:
            print(f"   Quality Score: {score}/100 - NEEDS IMPROVEMENT")
        else:
            print(f"   Quality Score: {score}/100 - POOR")

# Analyze chunking quality
analyze_chunking_quality(results)

# Overall summary
print(f"\n{'='*80}")
print("OVERALL SUMMARY")
print(f"{'='*80}")
print(f"Total documents processed: {len(results)}")
print(f"Total chunks created: {sum(len(doc_data['chunks']) for doc_data in results.values())}")
print("Chunking quality analysis complete!")

# Preview chunks for quality checking
print("\n" + "="*80)
print("Chunk Quality Preview")
print("="*80)

# Show preview of chunks from first document
if results:
    first_doc_name = list(results.keys())[0]
    first_doc = results[first_doc_name]
    
    print(f"\nPreviewing chunks from: {first_doc_name}")
    preview_chunks(first_doc['chunks'], num_samples=3)
    
    # Show preview from other documents if available
    if len(results) > 1:
        for doc_name in list(results.keys())[1:]:
            print(f"\n{'='*80}")
            print(f"Previewing chunks from: {doc_name}")
            preview_chunks(results[doc_name]['chunks'], num_samples=2)

# Step 3: Create embeddings and store in vector store
    print("\n" + "="*80)
    print("Step 3: Creating Embeddings & Vector Store")
    print("="*80)
    
    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Model loaded (embedding dimension: {embedding_model.get_sentence_embedding_dimension()})")
    
    # Collect all chunks with their metadata
    all_chunks = []
    chunk_texts = []
    
    print("\nCollecting chunks from all documents...")
    for doc_name, doc_data in results.items():
        for chunk in doc_data['chunks']:
            chunk_text = chunk.get('text', '')
            if chunk_text.strip():  # Only process non-empty chunks
                all_chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk.get('chunk_id', len(all_chunks)),
                    'filename': doc_name,
                    'metadata': chunk.get('metadata', {})
                })
                chunk_texts.append(chunk_text)
    
    total_chunks = len(chunk_texts)
    print(f"✓ Collected {total_chunks} chunks for embedding")
    
    # Create embeddings in batches
    print(f"\nCreating embeddings (this may take a few minutes)...")
    batch_size = 32
    embeddings = []
    
    for i in range(0, total_chunks, batch_size):
        batch = chunk_texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")
    
    embeddings = np.array(embeddings).astype('float32')
    print(f"✓ Created embeddings: shape {embeddings.shape}")
    
    # Create FAISS index
    print(f"\nCreating FAISS vector index...")
    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✓ FAISS index created with {index.ntotal} vectors")
    
    # Save vector store
    print(f"\nSaving vector store to {VECTOR_STORE_DIR}...")
    
    # Save FAISS index
    faiss_index_path = os.path.join(VECTOR_STORE_DIR, 'faiss_index.bin')
    faiss.write_index(index, faiss_index_path)
    print(f"✓ FAISS index saved to {faiss_index_path}")
    
    # Save chunk metadata
    metadata_path = os.path.join(VECTOR_STORE_DIR, 'chunk_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_chunks, f)
    print(f"✓ Chunk metadata saved to {metadata_path}")
    
    # Save embeddings info
    embeddings_info = {
        'model_name': EMBEDDING_MODEL,
        'dimension': dimension,
        'total_chunks': total_chunks,
        'chunks_per_document': {doc_name: len(doc_data['chunks']) for doc_name, doc_data in results.items()}
    }
    info_path = os.path.join(VECTOR_STORE_DIR, 'embeddings_info.json')
    with open(info_path, 'w') as f:
        json.dump(embeddings_info, f, indent=2)
    print(f"✓ Embeddings info saved to {info_path}")
    
    print(f"\n{'='*80}")
    print("Vector Store Summary")
    print(f"{'='*80}")
    print(f"Total chunks embedded: {total_chunks}")
    print(f"Embedding dimension: {dimension}")
    print(f"Model used: {EMBEDDING_MODEL}")
    print(f"Vector store location: {VECTOR_STORE_DIR}")
    print(f"✓ Vector store created successfully!")
    
    # Test the vector store with a sample query
    print(f"\n{'='*80}")
    print("Testing Vector Store")
    print(f"{'='*80}")
    test_query = "customer due diligence requirements"
    print(f"Test query: '{test_query}'")
    
    query_embedding = embedding_model.encode([test_query])[0].astype('float32')
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search for top 3 similar chunks
    k = 3
    distances, indices = index.search(query_embedding, k)
    
    print(f"\nTop {k} most similar chunks:")
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0]), 1):
        chunk = all_chunks[idx]
        print(f"\n{i}. Similarity distance: {distance:.4f}")
        print(f"   Document: {chunk['filename']}")
        print(f"   Chunk ID: {chunk['chunk_id']}")
        print(f"   Preview: {chunk['text'][:200]}...")
