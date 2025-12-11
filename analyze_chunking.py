import json
import os
from collections import Counter

def analyze_chunking_quality(chunked_file='chunks/chunked_documents.json'):
    """Analyze the quality of chunked documents."""
    
    with open(chunked_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("="*80)
    print("CHUNKING QUALITY ANALYSIS REPORT")
    print("="*80)
    
    all_issues = []
    
    for doc_name, doc_data in data.items():
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
        
        print(f"\nSize Distribution:")
        for range_name, count in size_ranges.items():
            percentage = (count / total_chunks * 100) if total_chunks > 0 else 0
            bar = int(percentage / 2)
            print(f"   {range_name:20s}: {count:4d} ({percentage:5.1f}%) {bar}")
        
        # Identify issues
        issues = []
        
        # Very small chunks
        very_small = [(i, s, chunks[i].get('text', '')[:80]) 
                      for i, s in enumerate(sizes) if s < 100]
        if very_small:
            issues.append(f" {len(very_small)} very small chunks (< 100 chars)")
            print(f"\nVery Small Chunks (< 100 chars): {len(very_small)}")
            for idx, size, preview in very_small[:5]:
                print(f"   Chunk {idx}: {size} chars - '{preview}...'")
        
        # Very large chunks
        very_large = [(i, s, chunks[i].get('text', '')[:80]) 
                      for i, s in enumerate(sizes) if s > 800]
        if very_large:
            issues.append(f"{len(very_large)} very large chunks (> 800 chars)")
            print(f"\nVery Large Chunks (> 800 chars): {len(very_large)}")
            for idx, size, preview in very_large[:5]:
                print(f"   Chunk {idx}: {size} chars - '{preview}...'")
        
        # Check chunk boundaries (mid-sentence breaks)
        boundary_issues = []
        for i in range(len(chunks) - 1):
            chunk1_text = chunks[i].get('text', '').strip()
            chunk2_text = chunks[i+1].get('text', '').strip()
            
            if chunk1_text and chunk2_text:
                # Check if chunk ends without sentence ending and next starts with capital
                last_char = chunk1_text[-1] if chunk1_text else ''
                first_char = chunk2_text[0] if chunk2_text else ''
                
                if last_char not in '.!?;:\n' and first_char.isupper() and last_char.isalnum():
                    boundary_issues.append(i)
        
        if boundary_issues:
            issues.append(f"{len(boundary_issues)} potential mid-sentence breaks")
            print(f"\nMid-Sentence Breaks: {len(boundary_issues)} detected")
            for idx in boundary_issues[:5]:
                chunk1 = chunks[idx].get('text', '')[-50:]
                chunk2 = chunks[idx+1].get('text', '')[:50]
                print(f"   Between chunks {idx}-{idx+1}: ...'{chunk1}' | '{chunk2}'...")
        else:
            print(f"\nChunk boundaries look good")
        
        # Check for table of contents
        toc_chunks = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            if ('CHAPTER' in text or 'CONTENTS' in text) and ('...' in text or text.count('.') > 5):
                toc_chunks.append(i)
        
        if toc_chunks:
            issues.append(f"{len(toc_chunks)} table of contents chunks detected")
            print(f"\nTable of Contents Chunks: {len(toc_chunks)}")
            for idx in toc_chunks[:3]:
                print(f"   Chunk {idx}: '{chunks[idx].get('text', '')[:100]}...'")
        
        # Check for empty or whitespace-only chunks
        empty_chunks = [i for i, chunk in enumerate(chunks) 
                       if not chunk.get('text', '').strip()]
        if empty_chunks:
            issues.append(f"{len(empty_chunks)} empty chunks")
            print(f"\nEmpty Chunks: {len(empty_chunks)}")
        
        # Overall assessment
        print(f"\n{'='*80}")
        print("Overall Assessment:")
        
        score = 100
        if len(very_small) > total_chunks * 0.05:  # More than 5% very small
            score -= 20
            print("Too many very small chunks")
        if len(very_large) > total_chunks * 0.05:  # More than 5% very large
            score -= 20
            print("Too many very large chunks")
        if avg_size < 300:
            score -= 15
            print("Average chunk size too small")
        if avg_size > 700:
            score -= 15
            print("Average chunk size too large")
        if boundary_issues:
            score -= 10
            print("Mid-sentence breaks detected")
        
        if score >= 80:
            print(f"Quality Score: {score}/100 - GOOD")
        elif score >= 60:
            print(f"Quality Score: {score}/100 - NEEDS IMPROVEMENT")
        else:
            print(f"Quality Score: {score}/100 - POOR")
        
        all_issues.extend(issues)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total documents analyzed: {len(data)}")
    print(f"Total chunks: {sum(len(doc_data['chunks']) for doc_data in data.values())}")
    
    if all_issues:
        print(f"\nIssues found:")
        for issue in set(all_issues):
            print(f"  {issue}")
    else:
        print("\nNo major issues detected!")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    analyze_chunking_quality()

