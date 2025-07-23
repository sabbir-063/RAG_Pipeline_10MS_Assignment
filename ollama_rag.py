from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils.pdf_loader import load_and_chunk_pdf
from utils.cleaner import clean_docs
from utils.embedder import get_embedder

# Custom prompt template that handles both Bangla and English
PROMPT_TEMPLATE = """You are a helpful assistant that can answer questions in both Bangla and English.
If the question is in Bangla, answer in Bangla. If the question is in English, answer in English.

Context information is below:
-------------------
{context}
-------------------

Given the context information, answer the following question. 
If you cannot find the answer in the context, say "দুঃখিত, আমি এই প্রশ্নের উত্তর খুঁজে পাচ্ছি না।" for Bangla questions
or "Sorry, I cannot find the answer to this question." for English questions.

Question: {question}

Answer: """

def build_vectorstore():
    chunks = load_and_chunk_pdf("data/HSC26-Bangla1st-Paper.pdf")
    clean_chunks = clean_docs(chunks)

    embedder = get_embedder()
    vectordb = Chroma.from_documents(
        clean_chunks, 
        embedding=embedder, 
        persist_directory="vector_store"
    )
    return vectordb

def get_qa_chain():
    vectordb = Chroma(
        embedding_function=get_embedder(),
        persist_directory="vector_store"
    )
    # Increase k to get more context
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = OllamaLLM(
        model="gemma3",
        temperature=0.1  # Lower temperature for more focused answers
    )

    # Create custom prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # This combines all context into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
            "verbose": True
        }
    )
    return rag_chain

if __name__ == "__main__":
    # Build once
    build_vectorstore()

    qa_chain = get_qa_chain()

    while True:
        query = input("\n🤖 Ask me (Bangla or English): ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        # Use invoke instead of calling directly
        result = qa_chain.invoke({"query": query})
        print("\n📘 Answer: ", result["result"])
        
        # Print sources for debugging
        print("\n📚 Sources:")
        for idx, doc in enumerate(result["source_documents"], 1):
            print(f"\nSource {idx}:")
            print(doc.page_content[:200], "...")
