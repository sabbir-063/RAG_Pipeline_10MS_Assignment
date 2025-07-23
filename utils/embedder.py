import os
from langchain_huggingface import HuggingFaceEmbeddings

# Set Hugging Face API key
os.environ["HUGGINGFACE_API_KEY"] = "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"

def get_embedder():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},  # or "cuda"
        encode_kwargs={"normalize_embeddings": True}
    )
