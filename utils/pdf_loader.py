from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

def is_mcq_or_answer(text):
    # Pattern to detect MCQ options or answer explanations
    mcq_patterns = [
        r'(?:[কখগঘঙ]\.|\([কখগঘঙ]\))',  # Bangla MCQ options
        r'(?:[abcd]\.|\([abcd]\))',  # English MCQ options
        r'উত্তর[:]?',  # Answer indicator
        r'ব্যাখ্যা[:]?',  # Explanation indicator
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in mcq_patterns)

def split_by_content_type(text):
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    
    content_blocks = []
    current_block = []
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        if is_mcq_or_answer(para):
            # If we have a previous block, save it
            if current_block:
                content_blocks.append('\n'.join(current_block))
                current_block = []
            # Save MCQ/Answer as its own block
            content_blocks.append(para)
        else:
            current_block.append(para)
    
    # Add any remaining block
    if current_block:
        content_blocks.append('\n'.join(current_block))
    
    return content_blocks

def load_and_chunk_pdf(pdf_path, chunk_size=800, chunk_overlap=100):
    # Use PyPDFLoader for better Bangla support
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    
    processed_docs = []
    for doc in raw_docs:
        # Split content by type (passages vs MCQs)
        content_blocks = split_by_content_type(doc.page_content)
        
        for block in content_blocks:
            if block.strip():
                # Create a new document with the same metadata
                new_doc = doc.copy()
                new_doc.page_content = block
                processed_docs.append(new_doc)
    
    # Use a text splitter configured for Bangla
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "।", ".", " ", ""],  # Added Bangla full stop
        keep_separator=True,
        length_function=len
    )
    
    chunks = splitter.split_documents(processed_docs)
    return chunks
