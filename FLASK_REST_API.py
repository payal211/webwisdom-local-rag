from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import sys
import traceback
from typing import Dict, Any
import time

# Import the RAG system (assuming rag.py is in the same directory)
try:
    from rag import RAGSystem
except ImportError as e:
    print(f"Error importing RAG system: {e}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system with default models"""
    global rag_system
    try:
        logger.info("Initializing RAG system...")
        # Fixed: Only pass embedding_model_name since RAGSystem doesn't accept llm_model_name
        rag_system = RAGSystem(
            embedding_model_name="sentence-transformers/all-MiniLM-L12-v2"
        )
        logger.info("RAG system initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        return False

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "RAG API is running",
        "endpoints": {
            "health": "/",
            "load_content": "/load_content",
            "query": "/query",
            "query_batch": "/query_batch",
            "chunks": "/chunks",
            "status": "/status"
        }
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get current status of the RAG system"""
    if rag_system is None:
        return jsonify({
            "initialized": False,
            "error": "RAG system not initialized"
        }), 500
    
    return jsonify({
        "initialized": True,
        "content_loaded": len(rag_system.chunks) > 0,
        "num_chunks": len(rag_system.chunks),
        "embeddings_created": rag_system.embeddings is not None,
        "source_url": rag_system.source_url if hasattr(rag_system, 'source_url') else None,
        "embedding_model": rag_system.embedding_model_name,
        # Removed llm_model reference since it doesn't exist in your RAGSystem
        "system_info": rag_system.get_system_info() if hasattr(rag_system, 'get_system_info') else {}
    })

@app.route('/load_content', methods=['POST'])
def load_content():
    """Load content from a web URL"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "error": "URL is required in request body"
            }), 400
        
        url = data['url']
        chunk_size = data.get('chunk_size', 500)
        overlap = data.get('overlap', 50)
        
        logger.info(f"Loading content from URL: {url}")
        
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return jsonify({
                "error": "Invalid URL format. Must start with http:// or https://"
            }), 400
        
        # Load content
        start_time = time.time()
        content = rag_system.load_web_content(url)
        
        # Split into chunks - using the correct method name from your RAGSystem
        chunks = rag_system.smart_text_splitting(content, chunk_size=chunk_size, overlap=overlap)
        
        # Create embeddings
        embeddings = rag_system.create_embeddings()
        
        processing_time = time.time() - start_time
        
        logger.info(f"Content loaded successfully from {url}")
        
        return jsonify({
            "success": True,
            "message": "Content loaded and processed successfully",
            "data": {
                "url": url,
                "content_length": len(content),
                "num_chunks": len(chunks),
                "embedding_shape": embeddings.shape if embeddings is not None else "Unknown",
                "chunk_size": chunk_size,
                "overlap": overlap,
                "processing_time": round(processing_time, 2)
            }
        })
        
    except Exception as e:
        logger.error(f"Error loading content: {str(e)}")
        return jsonify({
            "error": f"Failed to load content: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/query', methods=['POST'])
def query():
    """Main query endpoint for the RAG system"""
    try:
        # Check if RAG system is initialized
        if rag_system is None:
            return jsonify({
                "error": "RAG system not initialized"
            }), 500
        
        # Check if content is loaded
        if len(rag_system.chunks) == 0:
            return jsonify({
                "error": "No content loaded. Please load content first using /load_content endpoint"
            }), 400
        
        # Get request data
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                "error": "Question is required in request body"
            }), 400
        
        question = data['question']
        top_k = data.get('top_k', 3)
        threshold = data.get('threshold', 0.1)
        
        if not question.strip():
            return jsonify({
                "error": "Question cannot be empty"
            }), 400
        
        logger.info(f"Processing query: {question}")
        logger.info(f"Top K: {top_k}, Threshold: {threshold}")
        
        # Process query using the correct method signature from your RAGSystem
        start_time = time.time()
        result = rag_system.query(question, top_k=top_k, threshold=threshold)
        processing_time = time.time() - start_time
        
        # Your RAGSystem.query method returns a dict with comprehensive information
        if isinstance(result, dict):
            answer = result.get('answer', 'No answer generated')
            chunks_used = result.get('chunks_used', [])
            confidence = result.get('confidence', 0.0)
            method = result.get('method', 'Unknown')
        else:
            # Fallback if result is not a dict
            answer = str(result)
            chunks_used = []
            confidence = 0.0
            method = 'Unknown'
        
        # Format chunks for response
        formatted_chunks = []
        for i, chunk in enumerate(chunks_used):
            if isinstance(chunk, dict):
                formatted_chunks.append({
                    "chunk_index": chunk.get('chunk_index', i),
                    "text": chunk.get('chunk_text', ''),
                    "chunk_preview": chunk.get('preview', chunk.get('chunk_text', '')[:200] + "..."),
                    "similarity_score": chunk.get('similarity_score', 0.0)
                })
            else:
                # If chunk is just text
                formatted_chunks.append({
                    "chunk_index": i,
                    "text": str(chunk),
                    "chunk_preview": str(chunk)[:200] + "...",
                    "similarity_score": 0.0
                })
        
        response = {
            "success": True,
            "answer": answer,
            "chunks_used": formatted_chunks,
            "metadata": {
                "processing_time": round(processing_time, 2),
                "confidence": confidence,
                "method": method,
                "top_k": top_k,
                "threshold": threshold,
                "question": question,
                "num_chunks_used": len(formatted_chunks)
            }
        }
        
        logger.info(f"Query processed successfully in {processing_time:.2f} seconds")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            "error": f"Failed to process query: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/query_batch', methods=['POST'])
def query_batch():
    """Batch query endpoint for multiple questions"""
    try:
        # Check if RAG system is initialized
        if rag_system is None:
            return jsonify({
                "error": "RAG system not initialized"
            }), 500
        
        # Check if content is loaded
        if len(rag_system.chunks) == 0:
            return jsonify({
                "error": "No content loaded. Please load content first using /load_content endpoint"
            }), 400
        
        # Get request data
        data = request.get_json()
        if not data or 'questions' not in data:
            return jsonify({
                "error": "Questions array is required in request body"
            }), 400
        
        questions = data['questions']
        top_k = data.get('top_k', 3)
        threshold = data.get('threshold', 0.1)
        
        if not isinstance(questions, list) or len(questions) == 0:
            return jsonify({
                "error": "Questions must be a non-empty array"
            }), 400
        
        logger.info(f"Processing batch query with {len(questions)} questions")
        
        # Process each question
        start_time = time.time()
        results = []
        
        for i, question in enumerate(questions):
            if not question.strip():
                results.append({
                    "question": question,
                    "error": "Question cannot be empty"
                })
                continue
            
            try:
                result = rag_system.query(question, top_k=top_k, threshold=threshold)
                
                if isinstance(result, dict):
                    answer = result.get('answer', 'No answer generated')
                    chunks_used = result.get('chunks_used', [])
                    confidence = result.get('confidence', 0.0)
                else:
                    answer = str(result)
                    chunks_used = []
                    confidence = 0.0
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "chunks_used": len(chunks_used),
                    "confidence": confidence
                })
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                results.append({
                    "question": question,
                    "error": str(e)
                })
        
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "results": results,
            "metadata": {
                "processing_time": round(processing_time, 2),
                "questions_processed": len(questions),
                "top_k": top_k,
                "threshold": threshold
            }
        }
        
        logger.info(f"Batch query processed successfully in {processing_time:.2f} seconds")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing batch query: {str(e)}")
        return jsonify({
            "error": f"Failed to process batch query: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/chunks', methods=['GET'])
def get_chunks():
    """Get chunks with pagination"""
    try:
        # Check if RAG system is initialized
        if rag_system is None:
            return jsonify({
                "error": "RAG system not initialized"
            }), 500
        
        # Check if content is loaded
        if len(rag_system.chunks) == 0:
            return jsonify({
                "error": "No content loaded. Please load content first using /load_content endpoint"
            }), 400
        
        # Get pagination parameters
        limit = int(request.args.get('limit', 10))
        offset = int(request.args.get('offset', 0))
        
        # Validate parameters
        if limit < 1 or limit > 100:
            return jsonify({
                "error": "Limit must be between 1 and 100"
            }), 400
        
        if offset < 0:
            return jsonify({
                "error": "Offset must be non-negative"
            }), 400
        
        # Get paginated chunks
        total_chunks = len(rag_system.chunks)
        chunks_slice = rag_system.chunks[offset:offset+limit]
        
        # Format chunks for response
        formatted_chunks = []
        for i, chunk in enumerate(chunks_slice):
            formatted_chunks.append({
                "index": offset + i,
                "text": chunk,
                "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "length": len(chunk),
                "word_count": len(chunk.split())
            })
        
        response = {
            "success": True,
            "chunks": formatted_chunks,
            "pagination": {
                "total_chunks": total_chunks,
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total_chunks,
                "has_previous": offset > 0
            }
        }
        
        return jsonify(response)
        
    except ValueError as e:
        return jsonify({
            "error": f"Invalid parameter: {str(e)}"
        }), 400
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return jsonify({
            "error": f"Failed to retrieve chunks: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/system_info', methods=['GET'])
def get_system_info():
    """Get comprehensive system information"""
    try:
        if rag_system is None:
            return jsonify({
                "error": "RAG system not initialized"
            }), 500
        
        info = rag_system.get_system_info()
        return jsonify({
            "success": True,
            "system_info": info
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({
            "error": f"Failed to get system info: {str(e)}"
        }), 500

@app.route('/chunk_stats', methods=['GET'])
def get_chunk_stats():
    """Get statistics about chunks"""
    try:
        if rag_system is None:
            return jsonify({
                "error": "RAG system not initialized"
            }), 500
        
        stats = rag_system.get_chunk_stats()
        return jsonify({
            "success": True,
            "chunk_stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting chunk stats: {str(e)}")
        return jsonify({
            "error": f"Failed to get chunk stats: {str(e)}"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/",
            "/status",
            "/load_content",
            "/query",
            "/query_batch",
            "/chunks",
            "/system_info",
            "/chunk_stats"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "Something went wrong on the server"
    }), 500

if __name__ == '__main__':
    # Initialize RAG system on startup
    if initialize_rag_system():
        logger.info("Starting Flask API server...")
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 5000)),
            debug=True
        )
    else:
        logger.error("Failed to initialize RAG system. Server not started.")
        sys.exit(1)