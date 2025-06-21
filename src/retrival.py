#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.generator import GroqGenerator

# Load environment variables
load_dotenv()

# — PROMPT FOR LEGAL DOCUMENT DATA —
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a legal AI assistant specializing in Terms & Conditions, Privacy Policies, and Legal Contracts. 

Use **only** the following context extracted from legal documents:
{context}

Provide accurate, precise answers based ONLY on the provided context. If information is not in the context, clearly state this. Use formal, professional language appropriate for legal documents. Always cite sources where relevant.

Question: {question}
Answer:"""
)

# — EMBEDDING MODEL & VECTORSTORE —
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# Check if vectorstore exists
if not os.path.exists("vectordb"):
    raise FileNotFoundError("Vectorstore not found. Please run 'python create_vectordb.py' to create it first.")

try:
    vectorstore = FAISS.load_local(
        "vectordb",
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"Error loading vectorstore: {e}")
    try:
        vectorstore = FAISS.load_local("vectordb", embedding_model)
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        raise ValueError("Could not load vectorstore. Please run 'python create_vectordb.py' to recreate it.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize the Groq generator
llm = GroqGenerator()

# — BUILD THE QA CHAIN (without conversational memory) —
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": qa_prompt}
)

def response(user_input: str) -> str:
    """Run the QA chain and return an answer."""
    try:
        result = qa_chain({"query": user_input})
        return result["result"]
    except Exception as e:
        return f"Error processing query: {str(e)}"

def response_with_sources(user_input: str) -> dict:
    """Run the QA chain and return answer with sources."""
    try:
        result = qa_chain({"query": user_input})
        return {
            "answer": result["result"],
            "sources": [doc.page_content[:200] + "..." for doc in result["source_documents"]]
        }
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": []
        }

def response_with_sources_streaming(user_input: str):
    """Run the QA chain and return streaming answer with sources."""
    try:
        # Get relevant documents first
        docs = retriever.invoke(user_input)
        
        # Prepare context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format prompt
        prompt = qa_prompt.format(context=context, question=user_input)
        
        # Initialize generator for streaming
        llm_instance = GroqGenerator()
        
        # Return sources immediately
        sources = [doc.page_content[:200] + "..." for doc in docs]
        
        # Stream the response
        response_stream = llm_instance.stream_call(prompt)
        
        return {
            "response_stream": response_stream,
            "sources": sources
        }
        
    except Exception as e:
        def error_stream():
            yield f"Error processing query: {str(e)}"
        
        return {
            "response_stream": error_stream(),
            "sources": []
        }
