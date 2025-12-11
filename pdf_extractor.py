import pymupdf  
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import json
import glob
import unicodedata

def deep_clean_text(text: str) -> str:
    """
    Deeply clean extracted PDF text by removing:
    - Non-printable characters
    - Excessive whitespace
    - Control characters
    - Special formatting artifacts
    - Fix common encoding issues
    """
    if not text:
        return ""
    
    # Remove non-printable characters and control characters (except newlines, tabs, spaces)
    # Keep only printable characters, newlines, tabs, and spaces
    cleaned = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t ')
    
    # Normalize unicode characters (e.g., convert smart quotes to regular quotes)
    cleaned = unicodedata.normalize('NFKD', cleaned)
    
    # Remove zero-width spaces and other invisible characters
    cleaned = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', cleaned)
    
    # Fix common hyphenation issues (words split across lines)
    # Pattern: word-\nword -> wordword (but be careful not to break real hyphens)
    cleaned = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', cleaned)
    
    # Remove excessive line breaks (more than 2 consecutive)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    # Remove excessive spaces (more than 2 consecutive)
    cleaned = re.sub(r' {3,}', ' ', cleaned)
    
    # Remove tabs and replace with single space
    cleaned = cleaned.replace('\t', ' ')
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in cleaned.split('\n')]
    cleaned = '\n'.join(lines)
    
    # Remove empty lines at the beginning and end
    cleaned = cleaned.strip()
    
    # Remove page numbers and common header/footer patterns
    # Pattern: standalone numbers (likely page numbers)
    cleaned = re.sub(r'^\d+$', '', cleaned, flags=re.MULTILINE)
    
    # Remove common PDF artifacts like "Page X of Y"
    cleaned = re.sub(r'Page\s+\d+\s+of\s+\d+', '', cleaned, flags=re.IGNORECASE)
    
    # Normalize spaces within lines (but preserve newlines)
    # Replace multiple spaces with single space, but keep newlines
    lines = cleaned.split('\n')
    lines = [re.sub(r' +', ' ', line) for line in lines]
    cleaned = '\n'.join(lines)
    
    # Remove multiple consecutive newlines again (keep max 2 for paragraph breaks)
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    
    # Final cleanup: remove any remaining control characters
    cleaned = ''.join(char for char in cleaned if char.isprintable() or char in '\n\t ')
    
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()
    
    return cleaned

# Get all PDF files in the documents folder
documents_folder = "/Users/krishnadahale/Documents/CGAIF/Capstone Project/documents"
pdf_files = glob.glob(os.path.join(documents_folder, "*.pdf"))

# Extract text from all PDFs in the folder
all_extracted_data = []

for pdf_path in pdf_files:
    print(f"Processing: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    
    # print the number of pages in the pdf
    print(f"Number of pages: {len(doc)}")
    
    # Extract text from all pages
    text = ""
    for page in doc:
        text += page.get_text("text")
    
    metadata = doc.metadata
    print(f"Metadata: {metadata}")
    
    # Deep clean the extracted text
    cleaned_text = deep_clean_text(text)
    sections = re.findall(r'Section\s+\d+', cleaned_text)
    print(f"Found {len(sections)} sections")
    
    # Store extracted data for this PDF
    all_extracted_data.append({
        'filename': os.path.basename(pdf_path),
        'text': cleaned_text,
        'sections': sections,
        'metadata': metadata,
        'page_count': len(doc)
    })
    
    doc.close()

# save the extracted and cleaned text to a json file
with open('extracted_text.json', 'w') as f:
    json.dump(all_extracted_data, f, indent=2)

print(f"\nExtracted text from {len(all_extracted_data)} PDF file(s) and saved to extracted_text.json")





