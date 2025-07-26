from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- CONFIG ---
EMBED_MODEL = "intfloat/multilingual-e5-base"
MODEL_NAME = "google/mt5-small"
VECTORSTORE_PATH_BN = "vectorstore"

# --- LOAD VECTORSTORE ---
def load_vectorstore(path):
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)

# --- QUERY VECTORSTORE ---
def query_vectorstore(query, db, top_k=3):
    results = db.similarity_search("query: " + query, k=top_k)
    return [doc.page_content for doc in results]

# --- GENERATE ANSWER ---
def generate_answer(query, context):
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    generator = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer
    )
    prompt = f"Answer the question using the context.\n\nQuestion: {query}\n\nContext: {context}"
    output = generator(prompt, max_new_tokens=256)[0]["generated_text"]
    return output

if __name__ == "__main__":
    db = load_vectorstore(VECTORSTORE_PATH_BN)
    print("Bangla RAG system ready. Type your question in Bangla (or 'exit' to quit):")
    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break
        docs = query_vectorstore(query, db)
        context = " ".join(docs)
        answer = generate_answer(query, context)
        print("Assistant:", answer)

# if __name__ == "__main__":
#     rag = MultilingualRAG()
    
#     # Test with sample questions
#     questions = [
#         "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
#         "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
#         "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
#         "Who is called handsome in Anupam's words?"
#     ]
    
#     for q in questions:
#         print(f"Q: {q}")
#         response = rag.query(q)
#         print(f"A: {response.get('answer', 'No answer')}")
#         print(f"Sources: {response.get('sources', [])[:1]}\n")