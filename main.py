from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import uvicorn
import os

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

# --- FASTAPI APP ---
app = FastAPI()
db = load_vectorstore(VECTORSTORE_PATH_BN)

templates = Jinja2Templates(directory="templates")
if not os.path.exists("templates"):
    os.makedirs("templates")
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write("""
<!doctype html>
<title>Bangla RAG Chatbot</title>
<h2>Bangla RAG Chatbot</h2>
<form method="post">
  <input name="query" style="width:400px" placeholder="Type your Bangla question here">
  <input type="submit" value="Ask">
</form>
{% if answer %}
  <h4>Answer:</h4>
  <div style="white-space: pre-wrap;">{{ answer }}</div>
{% endif %}
""")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "answer": None})

@app.post("/", response_class=HTMLResponse)
async def home_post(request: Request, query: str = Form(...)):
    docs = query_vectorstore(query, db)
    context = " ".join(docs)
    answer = generate_answer(query, context)
    return templates.TemplateResponse("index.html", {"request": request, "answer": answer})

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    docs = query_vectorstore(request.query, db)
    context = " ".join(docs)
    answer = generate_answer(request.query, context)
    return {"answer": answer}

# Optional: for direct run (not needed for uvicorn)
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)