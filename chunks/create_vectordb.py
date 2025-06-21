import os
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from nltk.tokenize import sent_tokenize
import nltk

def sentence_aware_chunking(text, chunk_size=300):
    """Create sentence-aware chunks from text."""
    chunks, current_chunk, word_count = [], [], 0
    for sentence in sent_tokenize(text):
        sentence_words = len(sentence.split())
        if word_count + sentence_words > chunk_size and current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk), 
                'word_count': word_count, 
                'chunk_id': len(chunks)
            })
            current_chunk, word_count = [], 0
        current_chunk.append(sentence)
        word_count += sentence_words
    
    if current_chunk:
        chunks.append({
            'text': ' '.join(current_chunk), 
            'word_count': word_count, 
            'chunk_id': len(chunks)
        })
    return chunks

def create_vectorstore():
    """Create a new FAISS vectorstore from the document."""
    # Load and preprocess the document
    text_path = os.path.join("data", "AI Training Document.pdf")
    
    # Check if preprocessed text exists
    preprocessed_path = os.path.join("data", "preprocessed_AI Training Document.txt")
    
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed text from {preprocessed_path}")
        with open(preprocessed_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print(f"Preprocessed text not found at {preprocessed_path}")
        print("Please create the preprocessed text file first or update the path.")
        return
    
    print("Creating chunks...")
    chunks = sentence_aware_chunking(text, chunk_size=300)
    print(f"Created {len(chunks)} chunks")
    
    # Convert chunks to LangChain Documents
    documents = []
    for chunk in chunks:
        doc = Document(
            page_content=chunk['text'],
            metadata={
                'chunk_id': chunk['chunk_id'],
                'word_count': chunk['word_count'],
                'source': 'AI Training Document'
            }
        )
        documents.append(doc)
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    # Create FAISS vectorstore
    print("Creating FAISS vectorstore...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the vectorstore
    print("Saving vectorstore...")
    vectorstore.save_local("vectordb")
    
    print("Vectorstore created successfully!")
    
    # Test the vectorstore
    print("\nTesting vectorstore...")
    test_query = "how can a user terminate their contract?"
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"Test query: {test_query}")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result.page_content[:100]}...")

if __name__ == "__main__":
    create_vectorstore()
