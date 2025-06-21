# 🏛️ Legal Document AI Chatbot

An AI-powered chatbot capable of answering user queries based on legal documents (Terms & Conditions, Privacy Policies, Legal Contracts) using a Retrieval-Augmented Generation (RAG) pipeline with streaming responses.

## 🎯 Overview

This project implements a complete RAG pipeline using:
- **LLM**: Llama3-8B via Groq API for fast inference
- **Vector Database**: FAISS for semantic search
- **Embeddings**: BGE-large-en-v1.5 for high-quality document embeddings
- **Interface**: Streamlit with real-time streaming responses
- **Evaluation**: RAGAS metrics for RAG quality assessment

## 📁 Project Structure

```
labs/
├── 📄 data/                          # Document storage
│   ├── AI Training Document.pdf      # Source legal document
│   └── preprocessed_AI Training Document.txt  # Cleaned text
├── 🧱 chunks/                        # Legacy chunking utilities
│   └── create_vectordb.py            # Alternative vector DB creation
├── 🗄️ vectordb/                      # FAISS vector database
│   ├── index.faiss                   # Vector embeddings
│   └── index.pkl                     # Metadata
├── 📔 notebook/                      # Preprocessing and evaluation
│   ├── evaluater.py                  # RAGAS evaluation script
│   └── preprocessing.py              # PDF text extraction & cleaning
├── 🔧 src/                          # Core RAG components
│   ├── generator.py                  # Groq LLM integration
│   └── retrival.py                  # Main RAG pipeline with sources
├── 🌐 app.py                        # Streamlit web interface
├── 🛠️ create_vectordb.py            # Vector database creation (main)
├── 📋 requirements.txt               # Python dependencies
├── 🔐 .env                          # Environment variables
└── 📖 README.md                     # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone/download the project
cd labs

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create/update `.env` file with your Groq API key:
```bash
# Get your API key from: https://console.groq.com/keys
GROQ_API_KEY=your-groq-api-key-here
```

### 3. Prepare Vector Database

If vectordb doesn't exist or needs regeneration:
```bash
python create_vectordb.py
```

### 4. Run the Chatbot

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## 🔧 Architecture & Components

### Document Processing Pipeline

1. **PDF Extraction** (`notebook/preprocessing.py`)
   - Extracts text from PDF documents
   - Removes headers, footers, and formatting artifacts
   - Preserves document structure and legal formatting

2. **Text Chunking** (`create_vectordb.py`)
   - Sentence-aware chunking (100-300 words)
   - Maintains context boundaries
   - Preserves legal document semantics

3. **Embedding Generation**
   - Model: `BAAI/bge-large-en-v1.5`
   - High-quality multilingual embeddings
   - Optimized for legal/business text

### RAG Pipeline

#### Core Components:
- **Retriever**: FAISS similarity search (top-3 chunks)
- **Generator**: Groq Llama3-8B with legal-specific prompts
- **Pipeline**: Single RAG implementation with source tracking

#### Main Implementation:

**Primary RAG** (`src/retrival.py`)
- Clean, efficient design
- Source document tracking
- Response with citations
- Error handling and fallbacks

### Streamlit Interface

#### Features:
- 💬 **True token-by-token streaming responses**
- 📄 **Source document display** in expandable sections
- 🔄 **Clear chat functionality**
- 📊 **Model information sidebar**
- ⚡ **Real-time Groq API streaming**

#### UI Components:
- Chat input with legal document context
- Real token-by-token streaming display
- Source citation display
- Model and database statistics

## 🔍 Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM** | Groq Llama3-8B | Fast, high-quality text generation |
| **Embeddings** | BGE-large-en-v1.5 | Semantic document understanding |
| **Vector DB** | FAISS | Efficient similarity search |
| **Framework** | LangChain | RAG pipeline orchestration |
| **Interface** | Streamlit | Interactive web application |
| **Evaluation** | RAGAS | RAG quality metrics |

## 📊 Model Choices & Rationale

### LLM Selection: Groq Llama3-8B
- **Speed**: Sub-second inference times
- **Quality**: Strong instruction following
- **Cost**: Efficient API pricing
- **Legal Context**: Good understanding of formal language

### Embedding Model: BGE-large-en-v1.5
- **Performance**: MTEB leaderboard top performer
- **Multilingual**: Supports various document types
- **Size**: Good balance of quality vs. speed
- **Domain**: Excellent for business/legal text

### Vector Database: FAISS
- **Speed**: Optimized similarity search
- **Scalability**: Handles large document collections
- **Memory**: Efficient storage format
- **Integration**: Seamless LangChain compatibility

## 💡 Usage Examples

### Sample Queries:

1. **Privacy Policies**:
   ```
   "What data does the company collect from users?"
   "How long is personal data retained?"
   ```

2. **Terms & Conditions**:
   ```
   "What are the termination conditions?"
   "What payment methods are accepted?"
   ```

3. **Legal Contracts**:
   ```
   "What are the liability limitations?"
   "How are disputes resolved?"
   ```

### Expected Response Format:
```
Based on the provided legal context, [detailed answer with formal language].

Source references are automatically included and displayed in expandable sections below the response.
```

## 🧪 Evaluation & Testing

### RAGAS Metrics (`notebook/evaluater.py`):
- **Faithfulness**: Answer grounding in source documents
- **Answer Relevancy**: Response relevance to query
- **Context Precision**: Retrieved chunk quality
- **Context Recall**: Source coverage completeness

### Run Evaluation:
```bash
python notebook/evaluater.py
```
## 📹 Demo Video

[Watch the Demo Video]
https://drive.google.com/file/d/1NW8nX2k05ALY1qMVwazgxy0NsE0XHUyb/view?usp=sharing
