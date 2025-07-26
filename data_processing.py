import os
from typing import List
from langchain.schema import Document
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
EMBED_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
VECTORSTORE_PATH_BN = "vectorstore"

# --- LOADERS ---
def load_txt(path: str) -> List[Document]:
    return TextLoader(path, encoding="utf-8").load()

def load_pdf(path: str) -> List[Document]:
    return PyPDFLoader(path).load()

def load_documents(paths: List[str]) -> List[Document]:
    all_docs = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            all_docs.extend(load_txt(path))
        elif ext == ".pdf":
            all_docs.extend(load_pdf(path))
    return all_docs

# --- SPLITTERS ---
def recursive_split(docs: List[Document], chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[Document]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", "।"]
    )
    return splitter.split_documents(docs)

def paragraph_split(docs: List[Document], chunk_size=512) -> List[Document]:
    chunks = []
    for doc in docs:
        paras = doc.page_content.split("\n\n")
        buf = ""
        for para in paras:
            if len(buf) + len(para) <= chunk_size:
                buf += para + "\n\n"
            else:
                chunks.append(Document(page_content=buf.strip(), metadata=doc.metadata))
                buf = para + "\n\n"
        if buf:
            chunks.append(Document(page_content=buf.strip(), metadata=doc.metadata))
    return chunks

def semantic_split(docs: List[Document], threshold=0.70) -> List[Document]:
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    chunks = []
    for doc in docs:
        paras = doc.page_content.split("\n\n")
        if len(paras) < 2:
            chunks.append(doc)
            continue
        buf = []
        for i in range(len(paras) - 1):
            buf.append(paras[i])
            v1 = embedder.embed_query("passage: " + paras[i])
            v2 = embedder.embed_query("passage: " + paras[i + 1])
            sim = cosine_similarity([v1], [v2])[0][0]
            if sim < threshold:
                chunks.append(Document(page_content="\n\n".join(buf).strip(), metadata=doc.metadata))
                buf = []
        buf.append(paras[-1])
        if buf:
            chunks.append(Document(page_content="\n\n".join(buf).strip(), metadata=doc.metadata))
    return chunks

# --- VECTORSTORE ---
def save_vectorstore(docs: List[Document], path: str):
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.from_documents(docs, embedder)
    vectordb.save_local(path)
    print(f"Vectorstore saved to: {path}")

if __name__ == "__main__":
    # Example: process all Bangla PDFs in data/bangla
    bn_dir = "data/bangla"
    bn_files = [os.path.join(bn_dir, f) for f in os.listdir(bn_dir) if f.endswith(".pdf") or f.endswith(".txt")]
    docs = load_documents(bn_files)
    # Choose one: recursive_split, paragraph_split, or semantic_split
    chunks = semantic_split(docs)
    save_vectorstore(chunks, VECTORSTORE_PATH_BN)