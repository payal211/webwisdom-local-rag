import os
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Dict, Optional
import logging
import torch
import warnings
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import time
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gc

warnings.filterwarnings('ignore')

# Download NLTK data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGWithLLM:
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "microsoft/DialoGPT-medium",
                 use_local_llm: bool = True):
        """
        Initialize Enhanced RAG System with Local LLM Integration
        
        Args:
            embedding_model_name: Name of the sentence transformer model
            llm_model_name: Name of the local LLM model for answer generation
            use_local_llm: Whether to use local LLM for answer generation
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.use_local_llm = use_local_llm
        
        self.chunks = []
        self.embeddings = None
        self.embedding_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.llm_pipeline = None
        self.source_url = None
        self.metadata = {}
        self.chunk_metadata = []
        
        # Improved chunking parameters
        self.min_chunk_size = 100
        self.max_chunk_size = 800
        self.optimal_chunk_size = 400
        
        # Answer generation parameters
        self.max_answer_length = 300
        self.min_answer_length = 50
        
        self._load_models()
    
    def _load_models(self):
        """Load embedding model and optional LLM with better error handling"""
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Move to appropriate device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_model = self.embedding_model.to(device)
            
            logger.info("Embedding model loaded successfully")
            
            if self.use_local_llm:
                logger.info(f"Loading local LLM: {self.llm_model_name}")
                self._load_local_llm()
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _load_local_llm(self):
        """Load local LLM for answer generation"""
        try:
            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            
            # Set pad token if not exists
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Load model with appropriate settings for text generation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Try different model configurations based on model type
            if "gpt" in self.llm_model_name.lower():
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            else:
                # Fallback to a more reliable model
                self.llm_model_name = "distilgpt2"
                logger.info(f"Switching to more reliable model: {self.llm_model_name}")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
            
            self.llm_model = self.llm_model.to(device)
            
            # Create pipeline for easier text generation
            self.llm_pipeline = pipeline(
                "text-generation",
                model=self.llm_model,
                tokenizer=self.llm_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_full_text=False,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=150,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            
            logger.info("Local LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading local LLM: {str(e)}")
            logger.info("Falling back to extractive answering only")
            self.use_local_llm = False
    
    def load_web_content(self, url: str) -> str:
        """Enhanced web content loading with better text extraction"""
        try:
            logger.info(f"Loading content from: {url}")
            self.source_url = url
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
                element.decompose()
            
            # Focus on main content areas
            main_content = (soup.find('main') or 
                          soup.find('article') or 
                          soup.find('div', class_=re.compile(r'content|main|body', re.I)) or 
                          soup.find('div', id=re.compile(r'content|main|body', re.I)) or
                          soup)
            
            # Extract text with better formatting preservation
            text = main_content.get_text(separator=' ', strip=True)
            
            # Clean up text more carefully
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single
            text = text.strip()
            
            # Store metadata
            self.metadata = {
                'url': url,
                'title': soup.find('title').get_text() if soup.find('title') else 'Unknown',
                'content_length': len(text),
                'load_time': datetime.now().isoformat(),
                'word_count': len(text.split())
            }
            
            logger.info(f"Successfully loaded {len(text)} characters from URL")
            return text
            
        except Exception as e:
            logger.error(f"Error loading web content: {str(e)}")
            raise
    
    def improved_text_splitting(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """
        Improved text splitting with semantic awareness and better boundaries
        """
        logger.info(f"Improved splitting text into chunks (size: {chunk_size}, overlap: {overlap})")
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'[.!?]+\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        chunk_metadata = []
        current_chunk = ""
        current_sentences = []
        sentence_start_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would make chunk too long
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) > chunk_size and current_chunk:
                # Finalize current chunk
                chunks.append(current_chunk.strip())
                
                # Store metadata
                chunk_metadata.append({
                    'chunk_id': len(chunks) - 1,
                    'start_sentence': sentence_start_idx,
                    'end_sentence': i - 1,
                    'char_count': len(current_chunk.strip()),
                    'word_count': len(current_chunk.strip().split()),
                    'sentences': current_sentences.copy()
                })
                
                # Handle overlap - keep last few sentences
                if overlap > 0 and current_sentences:
                    overlap_sentences = current_sentences[-2:] if len(current_sentences) > 2 else current_sentences
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
                    sentence_start_idx = i
            else:
                current_chunk = potential_chunk
                current_sentences.append(sentence)
                if len(current_sentences) == 1:
                    sentence_start_idx = i
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            chunk_metadata.append({
                'chunk_id': len(chunks) - 1,
                'start_sentence': sentence_start_idx,
                'end_sentence': len(sentences) - 1,
                'char_count': len(current_chunk.strip()),
                'word_count': len(current_chunk.strip().split()),
                'sentences': current_sentences
            })
        
        # Filter chunks that are too small or too large
        filtered_chunks = []
        filtered_metadata = []
        for chunk, meta in zip(chunks, chunk_metadata):
            if self.min_chunk_size <= len(chunk) <= self.max_chunk_size:
                filtered_chunks.append(chunk)
                filtered_metadata.append(meta)
        
        self.chunks = filtered_chunks
        self.chunk_metadata = filtered_metadata
        logger.info(f"Created {len(filtered_chunks)} optimized chunks")
        return filtered_chunks
    
    def create_embeddings(self) -> np.ndarray:
        """Create embeddings with better normalization"""
        if not self.chunks:
            raise ValueError("No chunks available. Please load and split text first.")
        
        logger.info("Creating embeddings for chunks")
        
        # Create embeddings in batches
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            try:
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    batch_size=min(batch_size, len(batch)),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                all_embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i//batch_size}: {e}")
                raise
        
        self.embeddings = np.vstack(all_embeddings)
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def advanced_retrieval(self, query: str, top_k: int = 5, threshold: float = 0.2) -> List[Dict]:
        """
        Advanced retrieval with better ranking and deduplication
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not created. Please create embeddings first.")
        
        logger.info(f"Advanced retrieval for query: '{query}'")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get all chunks above threshold, sorted by similarity
        candidate_indices = [(i, score) for i, score in enumerate(similarities) if score > threshold]
        candidate_indices.sort(key=lambda x: x[1], reverse=True)
        
        # Apply diversity filtering to avoid redundant chunks
        selected_chunks = []
        selected_embeddings = []
        
        for idx, score in candidate_indices:
            if len(selected_chunks) >= top_k:
                break
            
            chunk_embedding = self.embeddings[idx:idx+1]
            
            # Check similarity with already selected chunks
            if selected_embeddings:
                similarities_to_selected = cosine_similarity(chunk_embedding, np.vstack(selected_embeddings))[0]
                if np.max(similarities_to_selected) > 0.8:  # Skip if too similar to selected
                    continue
            
            chunk_info = {
                'chunk_text': self.chunks[idx],
                'similarity_score': float(score),
                'chunk_index': int(idx),
                'metadata': self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {},
                'preview': self.chunks[idx][:150] + "..." if len(self.chunks[idx]) > 150 else self.chunks[idx]
            }
            
            selected_chunks.append(chunk_info)
            selected_embeddings.append(chunk_embedding)
        
        logger.info(f"Retrieved {len(selected_chunks)} diverse relevant chunks")
        return selected_chunks
    
    def generate_extractive_answer(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """
        Generate extractive answer (original RAG approach)
        """
        logger.info("Generating extractive answer")
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "chunks_used": [],
                "method": "Extractive - No relevant chunks found",
                "confidence": 0.0,
                "query": query
            }
        
        # Extract query keywords
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Find the most relevant sentences across chunks
        candidate_sentences = []
        
        for chunk_info in relevant_chunks[:3]:  # Focus on top 3 chunks
            chunk_text = chunk_info['chunk_text']
            chunk_score = chunk_info['similarity_score']
            
            # Split into sentences
            try:
                sentences = sent_tokenize(chunk_text)
            except:
                sentences = [s.strip() for s in chunk_text.split('.') if s.strip()]
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                sentence_lower = sentence.lower()
                sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
                
                # Calculate relevance score
                word_overlap = len(query_words.intersection(sentence_words))
                word_overlap_ratio = word_overlap / len(query_words) if query_words else 0
                
                relevance_score = chunk_score * 0.6 + word_overlap_ratio * 0.4
                
                candidate_sentences.append({
                    'sentence': sentence,
                    'score': relevance_score,
                    'chunk_index': chunk_info['chunk_index'],
                    'word_overlap': word_overlap,
                    'length': len(sentence)
                })
        
        # Sort by relevance score
        candidate_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        # Build answer from top sentences
        if candidate_sentences:
            selected_sentences = []
            total_length = 0
            
            for candidate in candidate_sentences[:3]:  # Max 3 sentences
                sentence = candidate['sentence']
                if total_length + len(sentence) <= self.max_answer_length:
                    selected_sentences.append(sentence)
                    total_length += len(sentence)
            
            answer = " ".join(selected_sentences)
            answer = re.sub(r'\s+', ' ', answer).strip()
            if not answer.endswith('.'):
                answer += "."
        else:
            answer = relevant_chunks[0]['chunk_text'][:200] + "..."
        
        # Calculate confidence
        if relevant_chunks:
            avg_similarity = np.mean([chunk['similarity_score'] for chunk in relevant_chunks[:3]])
            confidence = min(avg_similarity * 1.3, 1.0)
        else:
            confidence = 0.0
        
        return {
            "answer": answer,
            "chunks_used": relevant_chunks,
            "method": "Extractive Answer Generation",
            "confidence": float(confidence),
            "query": query,
            "answer_length": len(answer)
        }
    
    def generate_llm_enhanced_answer(self, query: str, relevant_chunks: List[Dict]) -> Dict:
        """
        Generate LLM-enhanced answer using local language model
        """
        logger.info("Generating LLM-enhanced answer")
        
        if not relevant_chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question.",
                "chunks_used": [],
                "method": "LLM Enhanced - No relevant chunks found",
                "confidence": 0.0,
                "query": query
            }
        
        if not self.llm_pipeline:
            logger.warning("LLM pipeline not available, falling back to extractive method")
            return self.generate_extractive_answer(query, relevant_chunks)
        
        try:
            # Prepare context from relevant chunks
            context_parts = []
            for chunk_info in relevant_chunks[:3]:  # Use top 3 chunks
                context_parts.append(chunk_info['chunk_text'][:300])  # Limit chunk size
            
            context = " ".join(context_parts)
            
            # Create prompt for the LLM
            prompt = f"""Based on the following context, please answer the question concisely and accurately.

Context: {context}

Question: {query}

Answer:"""
            
            # Generate response using the LLM
            try:
                # Generate with the pipeline
                response = self.llm_pipeline(
                    prompt,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                
                # Extract the generated text
                if response and len(response) > 0:
                    generated_text = response[0]['generated_text'].strip()
                    
                    # Clean up the generated text
                    generated_text = re.sub(r'\n+', ' ', generated_text)
                    generated_text = re.sub(r'\s+', ' ', generated_text).strip()
                    
                    # Limit length
                    if len(generated_text) > self.max_answer_length:
                        generated_text = generated_text[:self.max_answer_length].rsplit(' ', 1)[0] + "."
                    
                    answer = generated_text
                else:
                    raise Exception("No response generated")
                    
            except Exception as e:
                logger.error(f"Error in LLM generation: {e}")
                # Fallback to extractive method
                extractive_result = self.generate_extractive_answer(query, relevant_chunks)
                extractive_result['method'] = "LLM Enhanced (Fallback to Extractive)"
                return extractive_result
            
            # Calculate confidence based on relevance and length
            if relevant_chunks:
                avg_similarity = np.mean([chunk['similarity_score'] for chunk in relevant_chunks[:3]])
                length_score = min(len(answer) / 100, 1.0)  # Prefer longer, more complete answers
                confidence = min((avg_similarity * 0.7 + length_score * 0.3) * 1.2, 1.0)
            else:
                confidence = 0.0
            
            return {
                "answer": answer,
                "chunks_used": relevant_chunks,
                "method": "LLM Enhanced Answer Generation",
                "confidence": float(confidence),
                "query": query,
                "answer_length": len(answer),
                "context_used": context[:200] + "..." if len(context) > 200 else context
            }
            
        except Exception as e:
            logger.error(f"Error in LLM-enhanced generation: {str(e)}")
            # Fallback to extractive method
            extractive_result = self.generate_extractive_answer(query, relevant_chunks)
            extractive_result['method'] = "LLM Enhanced (Error - Fallback to Extractive)"
            extractive_result['error'] = str(e)
            return extractive_result
    
    def query_comparison(self, question: str, top_k: int = 5, threshold: float = 0.2) -> Dict:
        """
        Query with both methods and return comparison
        """
        try:
            start_time = time.time()
            
            # Input validation
            if not question or not question.strip():
                return {
                    "error": "Please provide a valid question.",
                    "processing_time": 0,
                    "query": question
                }
            
            # Use advanced retrieval
            relevant_chunks = self.advanced_retrieval(question, top_k=top_k, threshold=threshold)
            
            # Generate both types of answers
            extractive_result = self.generate_extractive_answer(question, relevant_chunks)
            llm_enhanced_result = self.generate_llm_enhanced_answer(question, relevant_chunks)
            
            processing_time = time.time() - start_time
            
            return {
                "query": question,
                "extractive_answer": extractive_result,
                "llm_enhanced_answer": llm_enhanced_result,
                "comparison": {
                    "extractive_length": len(extractive_result.get('answer', '')),
                    "llm_enhanced_length": len(llm_enhanced_result.get('answer', '')),
                    "extractive_confidence": extractive_result.get('confidence', 0.0),
                    "llm_enhanced_confidence": llm_enhanced_result.get('confidence', 0.0),
                    "chunks_used": len(relevant_chunks)
                },
                "processing_time": processing_time,
                "retrieval_params": {
                    "top_k": top_k,
                    "threshold": threshold,
                    "chunks_found": len(relevant_chunks)
                },
                "source_url": self.source_url,
                "embedding_model": self.embedding_model_name,
                "llm_model": self.llm_model_name if self.use_local_llm else "None"
            }
            
        except Exception as e:
            logger.error(f"Error processing query comparison: {str(e)}")
            return {
                "error": f"Error processing query: {str(e)}",
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                "query": question
            }
    
    def query(self, question: str, top_k: int = 5, threshold: float = 0.2, use_llm: bool = None) -> Dict:
        """
        Enhanced query method - can use either extractive or LLM-enhanced approach
        """
        if use_llm is None:
            use_llm = self.use_local_llm
        
        try:
            start_time = time.time()
            
            # Input validation
            if not question or not question.strip():
                return {
                    "answer": "Please provide a valid question.",
                    "chunks_used": [],
                    "method": "Input validation failed",
                    "processing_time": 0,
                    "query": question
                }
            
            # Use advanced retrieval
            relevant_chunks = self.advanced_retrieval(question, top_k=top_k, threshold=threshold)
            
            # Choose generation method
            if use_llm and self.llm_pipeline:
                result = self.generate_llm_enhanced_answer(question, relevant_chunks)
            else:
                result = self.generate_extractive_answer(question, relevant_chunks)
            
            # Add processing metadata
            result.update({
                "processing_time": time.time() - start_time,
                "source_url": self.source_url,
                "embedding_model": self.embedding_model_name,
                "llm_model": self.llm_model_name if use_llm else None,
                "retrieval_params": {
                    "top_k": top_k,
                    "threshold": threshold,
                    "chunks_found": len(relevant_chunks)
                }
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "chunks_used": [],
                "method": "Error occurred",
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                "query": question,
                "error": str(e)
            }
    
    def get_system_info(self) -> Dict:
        """Get comprehensive system information and status"""
        return {
            "embedding_model": self.embedding_model_name,
            "llm_model": self.llm_model_name if self.use_local_llm else "None",
            "llm_available": self.llm_pipeline is not None,
            "content_loaded": len(self.chunks) > 0,
            "num_chunks": len(self.chunks),
            "embeddings_created": self.embeddings is not None,
            "embedding_dimensions": self.embeddings.shape[1] if self.embeddings is not None else None,
            "source_url": self.source_url,
            "metadata": self.metadata,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "system_ready": self.embeddings is not None and len(self.chunks) > 0,
            "improvements": [
                "Advanced retrieval with diversity filtering",
                "Local LLM integration for enhanced answers",
                "Dual answer generation (extractive + generative)",
                "Better chunk boundaries",
                "Semantic deduplication",
                "Answer comparison capabilities"
            ]
        }
    
    def get_chunk_stats(self) -> Dict:
        """Get statistics about chunks"""
        if not self.chunks:
            return {"error": "No chunks available"}
        
        chunk_lengths = [len(chunk) for chunk in self.chunks]
        word_counts = [len(chunk.split()) for chunk in self.chunks]
        
        return {
            "total_chunks": len(self.chunks),
            "avg_chunk_length": np.mean(chunk_lengths),
            "median_chunk_length": np.median(chunk_lengths),
            "min_chunk_length": min(chunk_lengths),
            "max_chunk_length": max(chunk_lengths),
            "avg_words_per_chunk": np.mean(word_counts),
            "total_words": sum(word_counts),
            "total_characters": sum(chunk_lengths)
        }
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Additional utility functions for evaluation
def advanced_similarity_metrics(predicted: str, expected: str) -> Dict:
    """Advanced similarity metrics for better evaluation"""
    pred_words = set(re.findall(r'\b\w+\b', predicted.lower()))
    exp_words = set(re.findall(r'\b\w+\b', expected.lower()))
    
    if not exp_words:
        return {"error": "Expected answer is empty"}
    
    intersection = pred_words.intersection(exp_words)
    union = pred_words.union(exp_words)
    
    # Basic metrics
    jaccard = len(intersection) / len(union) if union else 0
    precision = len(intersection) / len(pred_words) if pred_words else 0
    recall = len(intersection) / len(exp_words) if exp_words else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    semantic_coverage = recall
    answer_relevance = precision
    length_ratio = len(predicted) / len(expected) if expected else 0
    
    return {
        "jaccard_similarity": jaccard,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "semantic_coverage": semantic_coverage,
        "answer_relevance": answer_relevance,
        "length_ratio": length_ratio,
        "word_overlap": len(intersection),
        "predicted_words": len(pred_words),
        "expected_words": len(exp_words)
    }

# Keep backward compatibility
RAGSystem = RAGWithLLM