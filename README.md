# Advanced RAG System with Local LLM Integration

A comprehensive Retrieval-Augmented Generation (RAG) system that answers questions based on web document content using locally hosted LLMs, with both REST API and Streamlit UI interfaces.

## 🚀 Features

- **Web Content Loading**: Extract and process content from any web URL
- **Intelligent Text Chunking**: Advanced text splitting with overlap for better context preservation
- **Vector Embeddings**: Uses sentence-transformers for high-quality semantic embeddings
- **Cosine Similarity Search**: Efficient vector similarity search with configurable thresholds
- **Dual Answer Generation**:
  - **Extractive Method**: Direct answer extraction from relevant chunks
  - **Local LLM Enhancement**: Natural language generation using local models
- **REST API**: FastAPI-based endpoints for programmatic access
- **Interactive UI**: Streamlit-based web interface with analytics dashboard
- **Docker Support**: Containerized deployment with docker-compose
- **Comprehensive Analytics**: Query history, performance metrics, and visualizations

## 📋 System Requirements

- Python 3.8+
- 4GB+ RAM (8GB+ recommended for larger models)
- CUDA-compatible GPU (optional, for faster inference)
- Docker and Docker Compose (for containerized deployment)

## 🔧 Installation & Setup

# Method 1: Docker Installation (Recommended)

## Download and extract the project files
## Navigate to the project directory
```bash
cd webpage-rag-system
```

## Build and run with Docker Compose
```bash
docker-compose build --no-cache
docker-compose up
```
## Access the applications

Streamlit UI: http://localhost:8501
REST API: http://localhost:5000

## 🚀 Quick Start
### Method 2: Local Installation

## Create conda environment
```bash
conda create -n rag_system python=3.9
conda activate rag_system
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the applications
```bash
# Terminal 1: Start API server
python FLASK_REST_API.py
```

```bash
# Terminal 2: Start Streamlit UI
streamlit run st_app.py
```
### Access Streamlit UI: http://localhost:8501
### Access REST API: http://localhost:5000

📊 Sample Outputs
Example 1: OWASP GenAI Security Query
Input URL: https://genai.owasp.org/initiatives/
Query: "What are the main AI security initiatives mentioned?"
Extractive Answer:
json{
  "answer": "The OWASP GenAI project focuses on AI security initiatives including threat modeling frameworks, security testing methodologies, and governance guidelines for organizations implementing AI systems.",
  "method": "extractive",
  "confidence": 0.87,
  "chunks_used": 3,
  "processing_time": 0.23
}
LLM Enhanced Answer:
json{
  "answer": "Based on the document, OWASP GenAI encompasses several comprehensive AI security initiatives. The main areas include developing threat modeling frameworks specifically for AI systems, creating security testing methodologies tailored to machine learning applications, and establishing governance guidelines to help organizations safely implement AI technologies while managing associated risks.",
  "method": "llm_enhanced", 
  "confidence": 0.92,
  "chunks_used": 3,
  "processing_time": 1.45
}
Example 2: Technical Documentation Query
Input URL: https://docs.python.org/3/tutorial/
Query: "How do you handle exceptions in Python?"
Extractive Answer:
json{
  "answer": "Python uses try-except blocks to handle exceptions. You can catch specific exceptions and handle them appropriately, or use a general except clause to catch all exceptions.",
  "method": "extractive",
  "confidence": 0.89,
  "chunks_used": 2,
  "processing_time": 0.18
}
Example 3: Comparison Mode Output
json{
  "extractive_answer": {
    "answer": "Machine learning models require careful validation and testing to ensure reliability.",
    "confidence": 0.82,
    "processing_time": 0.15
  },
  "llm_enhanced_answer": {
    "answer": "Machine learning models need comprehensive validation and testing procedures to ensure they perform reliably in production environments. This includes cross-validation, performance metrics evaluation, and robustness testing under various conditions.",
    "confidence": 0.88,
    "processing_time": 1.23
  },
  "comparison": {
    "better_method": "llm_enhanced",
    "confidence_difference": 0.06,
    "length_difference": 89
  }
}

## 📡 API Endpoints
Base URL: http://localhost:5000

MethodEndpointDescriptionGET/Health 
checkPOST/initializeInitialize 
system with
modelsPOST/load_contentLoad 
content from URLPOST/queryQuery the 
# systemPOST/query_comparisonCompare both methodsGET/system_infoGet system information
# Detailed API Documentation
# Initialize System
httpPOST /initialize
Content-Type: application/json

{
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "distilgpt2", 
    "use_local_llm": true
}
# Load Content
httpPOST /load_content
Content-Type: application/json

{
    "url": "https://example.com/article",
    "chunk_size": 400,
    "chunk_overlap": 50
}
# Query System
httpPOST /query
Content-Type: application/json

{
    "query": "What is this article about?",
    "top_k": 5,
    "threshold": 0.2,
    "use_llm": true
}

## 🤖 Supported Models
### Embedding Models

sentence-transformers/all-MiniLM-L6-v2 (Recommended - Fast & Efficient)
sentence-transformers/all-MiniLM-L12-v2 (Better Quality)
sentence-transformers/all-mpnet-base-v2 (Best Quality)

## Language Models

distilgpt2 (Recommended - Lightweight)
gpt2 (Standard GPT-2)
microsoft/DialoGPT-medium (Conversational)
microsoft/DialoGPT-small (Lightweight Conversational)

## 🔍 Technical Details
### Vector Similarity Method
The system uses Cosine Similarity for semantic search:

Formula: similarity = (A · B) / (||A|| × ||B||)
Range: 0 (no similarity) to 1 (identical)
Benefits: Handles variable document lengths, computationally efficient

## Text Processing Pipeline

Web Scraping: Extract clean text from HTML
Chunking: Split text with configurable size and overlap
Embedding: Convert chunks to dense vectors
Indexing: Store vectors for similarity search
Retrieval: Find relevant chunks using cosine similarity
Generation: Extract or generate answers using LLM

## 📊 Performance Metrics
### Typical Response Times

Extractive Method: 0.1 - 0.5 seconds
LLM Enhanced: 1.0 - 3.0 seconds
Content Loading: 2.0 - 10.0 seconds (depends on URL)

## Memory Usage

Base System: ~500MB
With Embeddings: ~1.5GB
With LLM: ~3.0GB

## 🛠️ Configuration Options
## Advanced Parameters
### python# Chunking Configuration
chunk_size = 400        # Characters per chunk (200-1000)
chunk_overlap = 50      # Overlap between chunks (0-200)

### Retrieval Configuration  
top_k = 5              # Number of chunks to retrieve (1-10)
threshold = 0.2        # Minimum similarity score (0.0-1.0)

### LLM Configuration
max_length = 200       # Maximum response length
temperature = 0.7      # Response creativity (0.1-1.0)

## 🐳 Docker Configuration
## Services

### rag-streamlit: Web UI on port 8501
### rag-api: REST API on port 5000

## Volumes

rag_data: Persistent data storage
rag_models: Model cache
rag_cache: HuggingFace model cache

## Commands
```bash
# Start services
docker-compose up

# Stop services  
docker-compose down

# View logs
docker-compose logs -f

# Rebuild
docker-compose build --no-cache


# ## 🔧 Troubleshooting
# Common Issues

# Port conflicts
```bash
# Check if ports are in use
netstat -an | grep -E "(5000|8501)"
```

## Memory issues

Reduce chunk_size and top_k parameters
Use smaller embedding models
Increase Docker memory limits


## Model download failures

Check internet connection
Clear model cache: rm -rf models/.cache
Try different model variants


## Web scraping issues

Some sites may block automated requests
Try different URLs
Check robots.txt compliance



## Performance Optimization

## Speed up responses

Use all-MiniLM-L6-v2 embedding model
Reduce top_k to 3-5
Use distilgpt2 for LLM


## Improve accuracy

Use all-mpnet-base-v2 embedding model
Increase top_k to 7-10
Fine-tune similarity threshold



📁 Project Structure
webpage-rag-system/
├── FLASK_REST_API.py      # REST API server
├── st_app.py              # Streamlit web interface  
├── rag.py                 # Core RAG implementation
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yaml   # Multi-container setup
├── README.md             # This documentation
├── .dockerignore         # Docker ignore patterns
└── data/                 # Data storage (created at runtime)
    └── models/           # Model cache (created at runtime)
## 🧪 Testing the System
#3# Manual Testing via Streamlit

Open http://localhost:8501
## Initialize system with default models
Load content from: https://genai.owasp.org/initiatives/
## Test queries:

"What is OWASP GenAI?"
"What are the main security concerns?"
"How can organizations implement AI safely?"



## API Testing via curl
```bash 
# Health check
curl http://localhost:5000/
```
## Full workflow test
curl -X POST http://localhost:5000/initialize -H "Content-Type: application/json" -d '{"embedding_model": "sentence-transformers/all-MiniLM-L6-v2", "llm_model": "distilgpt2", "use_local_llm": true}'

curl -X POST http://localhost:5000/load_content -H "Content-Type: application/json" -d '{"url": "https://genai.owasp.org/initiatives/"}'

curl -X POST http://localhost:5000/query -H "Content-Type: application/json" -d '{"query": "What is AI security?", "use_llm": true}'




Built with ❤️ for AI-powered question answering
