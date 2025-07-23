import re

def clean_bangla(text):
    # Handle common OCR errors in Bangla text
    replacements = {
        'ি': 'ি',  # Fix incorrect vowel sign
        'ে': 'ে',  # Fix incorrect vowel sign
        'া': 'া',  # Fix incorrect vowel sign
        '্': '্',  # Fix incorrect hasant
        '।।': '।',  # Fix multiple dari
        ',,': ',',  # Fix multiple commas
        '..': '.',  # Fix multiple dots
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Keep Bangla characters, English letters, numbers, and essential punctuation
    text = re.sub(r'[^\u0980-\u09FF\s\.\,\?\!\-a-zA-Z0-9।()]', ' ', text)
    
    # Fix extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s*([।\.\,\?\!\-\(\)])\s*', r'\1 ', text)
    
    return text.strip()

def clean_docs(docs):
    cleaned_docs = []
    for doc in docs:
        # Skip empty documents
        if not doc.page_content.strip():
            continue
            
        # Clean the text
        cleaned_text = clean_bangla(doc.page_content)
        
        # Skip if cleaned text is too short
        if len(cleaned_text) < 10:  # Minimum 10 characters
            continue
            
        doc.page_content = cleaned_text
        cleaned_docs.append(doc)
        
    return cleaned_docs
