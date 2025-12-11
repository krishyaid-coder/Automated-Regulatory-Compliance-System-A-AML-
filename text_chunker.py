"""
Text chunking utilities for regulatory documents
Simple functions for intelligent chunking
"""

from typing import List, Dict, Any, Optional
import re


def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentence_pattern = r'(?<=[.!?])\s+'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def split_by_sections(text: str) -> List[str]:
    """Split text by section markers (Section X, Chapter X, etc.)."""
    section_pattern = r'(?i)(?:Section|Chapter|Article|Paragraph)\s+\d+[\.:]?\s*\n'
    sections = re.split(section_pattern, text)
    
    if len(sections) <= 1:
        return [text]
    
    cleaned_sections = [s.strip() for s in sections if s.strip()]
    return cleaned_sections if len(cleaned_sections) > 1 else [text]


def chunk_by_size(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Chunk text by size with overlap, respecting sentence boundaries."""
    chunks = []
    sentences = split_sentences(text)
    
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'size': len(chunk_text)
            })
            
            # Start new chunk with overlap
            if chunk_overlap > 0:
                overlap_text = ' '.join(current_chunk[-3:])
                overlap_size = len(overlap_text)
                
                if overlap_size <= chunk_overlap:
                    current_chunk = current_chunk[-3:]
                    current_size = overlap_size
                else:
                    current_chunk = [sentence]
                    current_size = sentence_size
            else:
                current_chunk = [sentence]
                current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1
    
    # Add remaining chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append({
            'text': chunk_text,
            'size': len(chunk_text)
        })
    
    return chunks


def chunk_text(text: str, metadata: Optional[Dict] = None, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """Chunk text intelligently, respecting section boundaries."""
    if not text:
        return []
    
    chunks = []
    
    # Try to split by sections first
    sections = split_by_sections(text)
    
    if len(sections) > 1:
        # Chunk each section separately
        for section_text in sections:
            section_chunks = chunk_by_size(section_text, chunk_size, chunk_overlap)
            chunks.extend(section_chunks)
    else:
        # No clear sections, chunk by size
        chunks = chunk_by_size(text, chunk_size, chunk_overlap)
    
    # Add metadata to each chunk
    for i, chunk in enumerate(chunks):
        chunk['chunk_id'] = i
        chunk['chunk_index'] = i
        if metadata:
            chunk['metadata'] = metadata.copy()
        else:
            chunk['metadata'] = {}
    
    return chunks


def chunk_multiple_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Dict[str, Dict[str, Any]]:
    """Chunk multiple documents."""
    results = {}
    
    for doc in documents:
        filename = doc.get('filename', 'unknown')
        text = doc.get('text', '')
        metadata = {
            'filename': filename,
            'page_count': doc.get('page_count', 0),
            'sections': doc.get('sections', []),
            'original_metadata': doc.get('metadata', {})
        }
        
        chunks = chunk_text(text, metadata, chunk_size, chunk_overlap)
        
        results[filename] = {
            'filename': filename,
            'chunks': chunks,
            'num_chunks': len(chunks),
            'total_chars': len(text),
            'metadata': metadata
        }
    
    return results


def preview_chunks(chunks: List[Dict[str, Any]], num_samples: int = 3) -> None:
    """Preview sample chunks for quality checking."""
    if not chunks:
        print("No chunks to preview.")
        return
    
    print(f"\n{'='*80}")
    print(f"Chunk Preview (showing {min(num_samples, len(chunks))} of {len(chunks)} chunks)")
    print(f"{'='*80}\n")
    
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"Chunk {chunk.get('chunk_id', i)}:")
        print(f"  Size: {chunk.get('size', len(chunk.get('text', '')))} characters")
        print(f"  Text preview: {chunk.get('text', '')[:200]}...")
        if chunk.get('metadata'):
            print(f"  Metadata: {chunk['metadata']}")
        print()
