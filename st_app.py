import streamlit as st
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the RAG system
try:
    from rag import RAGWithLLM
except ImportError:
    st.error("Error: Could not import RAG system. Make sure rag.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Advanced RAG System with Local LLM",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .chunk-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #d0d0d0;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .method-comparison {
        display: flex;
        justify-content: space-between;
        margin: 1rem 0;
    }
    .method-box {
        flex: 1;
        margin: 0 0.5rem;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #d0d0d0;
    }
    .extractive-method {
        background-color: #fff2e6;
        border-color: #ff7f0e;
    }
    .llm-method {
        background-color: #e6f3ff;
        border-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'content_loaded' not in st.session_state:
    st.session_state.content_loaded = False
if 'embeddings_created' not in st.session_state:
    st.session_state.embeddings_created = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Title
st.markdown('<h1 class="main-header">🤖 Advanced RAG System with Local LLM Integration</h1>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.title("⚙️ Configuration")

# Model selection
st.sidebar.subheader("Embedding Model")
embedding_models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/paraphrase-MiniLM-L6-v2"
]
selected_embedding_model = st.sidebar.selectbox(
    "Choose embedding model:",
    embedding_models,
    index=0
)

st.sidebar.subheader("Local LLM Model")
llm_models = [
    "distilgpt2",
    "microsoft/DialoGPT-medium",
    "gpt2",
    "microsoft/DialoGPT-small"
]
selected_llm_model = st.sidebar.selectbox(
    "Choose local LLM model:",
    llm_models,
    index=0
)

use_local_llm = st.sidebar.checkbox("Enable Local LLM Integration", value=True)

# Advanced parameters
st.sidebar.subheader("Advanced Parameters")
chunk_size = st.sidebar.slider("Chunk Size", 200, 1000, 400, 50)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 200, 50, 10)
top_k = st.sidebar.slider("Top K Chunks", 1, 10, 5, 1)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.2, 0.05)

# Initialize RAG System
@st.cache_resource
def initialize_rag_system(embedding_model, llm_model, use_llm):
    """Initialize the RAG system with caching"""
    try:
        with st.spinner(f"Initializing RAG system with {embedding_model}..."):
            rag_system = RAGWithLLM(
                embedding_model_name=embedding_model,
                llm_model_name=llm_model,
                use_local_llm=use_llm
            )
        return rag_system, True
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        return None, False

# Load and display system status
if st.sidebar.button("Initialize System") or st.session_state.rag_system is None:
    rag_system, success = initialize_rag_system(selected_embedding_model, selected_llm_model, use_local_llm)
    if success:
        st.session_state.rag_system = rag_system
        st.sidebar.success("✅ RAG System initialized successfully!")
        
        # Display system info
        system_info = rag_system.get_system_info()
        with st.sidebar.expander("System Information"):
            st.json(system_info)
    else:
        st.sidebar.error("❌ Failed to initialize RAG system")

# Main content area
if st.session_state.rag_system is not None:
    rag_system = st.session_state.rag_system
    
    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Content Loading", "🔍 Query Interface", "📊 Analytics", "🔧 System Details"])
    
    with tab1:
        st.markdown('<h2 class="section-header">📄 Content Loading</h2>', unsafe_allow_html=True)
        
        # URL input
        url_input = st.text_input(
            "Enter URL to load content:",
            placeholder="https://example.com/article",
            help="Enter a valid URL to load and process content"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Load Content", type="primary"):
                if url_input:
                    try:
                        # Load content
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Loading web content...")
                        progress_bar.progress(25)
                        content = rag_system.load_web_content(url_input)
                        
                        status_text.text("Splitting content into chunks...")
                        progress_bar.progress(50)
                        chunks = rag_system.improved_text_splitting(
                            content, 
                            chunk_size=chunk_size, 
                            overlap=chunk_overlap
                        )
                        
                        status_text.text("Creating embeddings...")
                        progress_bar.progress(75)
                        embeddings = rag_system.create_embeddings()
                        
                        status_text.text("Content loaded successfully!")
                        progress_bar.progress(100)
                        
                        st.session_state.content_loaded = True
                        st.session_state.embeddings_created = True
                        
                        # Display success message
                        st.success(f"✅ Content loaded successfully!")
                        
                        # Display content statistics
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Total Chunks", len(chunks))
                        with col_b:
                            st.metric("Content Length", f"{len(content):,} chars")
                        with col_c:
                            st.metric("Embedding Dimensions", embeddings.shape[1])
                        
                        # Clear progress indicators
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                    except Exception as e:
                        st.error(f"❌ Error loading content: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a valid URL")
        
        with col2:
            if st.button("🗑️ Clear Content"):
                st.session_state.content_loaded = False
                st.session_state.embeddings_created = False
                st.session_state.query_history = []
                st.success("Content cleared!")
        
        # Display chunk statistics if content is loaded
        if st.session_state.content_loaded and hasattr(rag_system, 'chunks') and len(rag_system.chunks) > 0:
            st.markdown('<h3 class="section-header">📊 Chunk Statistics</h3>', unsafe_allow_html=True)
            
            chunk_stats = rag_system.get_chunk_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", chunk_stats.get('total_chunks', 0))
                st.metric("Min Length", f"{chunk_stats.get('min_chunk_length', 0):.0f}")
            with col2:
                st.metric("Avg Length", f"{chunk_stats.get('avg_chunk_length', 0):.0f}")
                st.metric("Max Length", f"{chunk_stats.get('max_chunk_length', 0):.0f}")
            with col3:
                st.metric("Total Words", f"{chunk_stats.get('total_words', 0):,}")
                st.metric("Avg Words/Chunk", f"{chunk_stats.get('avg_words_per_chunk', 0):.0f}")
            
            # Chunk length distribution
            if st.checkbox("Show Chunk Length Distribution"):
                chunk_lengths = [len(chunk) for chunk in rag_system.chunks]
                
                fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                
                # Histogram
                ax[0].hist(chunk_lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax[0].set_title('Chunk Length Distribution')
                ax[0].set_xlabel('Chunk Length (characters)')
                ax[0].set_ylabel('Frequency')
                
                # Box plot
                ax[1].boxplot(chunk_lengths, vert=True)
                ax[1].set_title('Chunk Length Box Plot')
                ax[1].set_ylabel('Chunk Length (characters)')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Preview chunks
            if st.checkbox("Preview Chunks"):
                num_preview = st.slider("Number of chunks to preview", 1, min(10, len(rag_system.chunks)), 3)
                
                for i in range(num_preview):
                    with st.expander(f"Chunk {i+1} ({len(rag_system.chunks[i])} chars)"):
                        st.text(rag_system.chunks[i])
    
    with tab2:
        st.markdown('<h2 class="section-header">🔍 Query Interface</h2>', unsafe_allow_html=True)
        
        if not st.session_state.content_loaded:
            st.warning("⚠️ Please load content first in the Content Loading tab")
        else:
            # Query input
            query_input = st.text_area(
                "Enter your question:",
                placeholder="What is this article about?",
                height=100,
                help="Enter your question about the loaded content"
            )
            
            # Query options
            col1, col2 = st.columns([2, 1])
            
            with col1:
                query_method = st.radio(
                    "Select query method:",
                    ["Both Methods (Comparison)", "Extractive Only", "LLM Enhanced Only"],
                    index=0
                )
            
            # with col2:
            #     st.write("**Vector Similarity Method:**")
            #     st.info("""
            #     **Cosine Similarity** is used for vector search:
            #     - Measures angle between query and chunk embeddings
            #     - Range: 0 (no similarity) to 1 (identical)
            #     - Robust to document length differences
            #     - Best for semantic similarity matching
            #     """)
            
            # Query button
            if st.button("🔍 Ask Question", type="primary"):
                if query_input.strip():
                    try:
                        start_time = time.time()
                        
                        # Execute query based on selected method
                        if query_method == "Both Methods (Comparison)":
                            result = rag_system.query_comparison(
                                query_input, 
                                top_k=top_k, 
                                threshold=similarity_threshold
                            )
                            
                            # Display both results
                            st.markdown('<h3 class="section-header">📋 Query Results Comparison</h3>', unsafe_allow_html=True)
                            
                            # Method comparison layout
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown('<div class="method-box extractive-method">', unsafe_allow_html=True)
                                st.markdown("**🔍 Extractive Method**")
                                extractive_result = result.get('extractive_answer', {})
                                st.markdown(f"**Answer:** {extractive_result.get('answer', 'No answer')}")
                                st.markdown(f"**Confidence:** {extractive_result.get('confidence', 0):.2f}")
                                st.markdown(f"**Method:** {extractive_result.get('method', 'Unknown')}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="method-box llm-method">', unsafe_allow_html=True)
                                st.markdown("**🤖 LLM Enhanced Method**")
                                llm_result = result.get('llm_enhanced_answer', {})
                                st.markdown(f"**Answer:** {llm_result.get('answer', 'No answer')}")
                                st.markdown(f"**Confidence:** {llm_result.get('confidence', 0):.2f}")
                                st.markdown(f"**Method:** {llm_result.get('method', 'Unknown')}")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Comparison metrics
                            comparison = result.get('comparison', {})
                            st.markdown('<h4>📊 Comparison Metrics</h4>', unsafe_allow_html=True)
                            
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Extractive Length", comparison.get('extractive_length', 0))
                                st.metric("LLM Enhanced Length", comparison.get('llm_enhanced_length', 0))
                            with metric_col2:
                                st.metric("Extractive Confidence", f"{comparison.get('extractive_confidence', 0):.2f}")
                                st.metric("LLM Enhanced Confidence", f"{comparison.get('llm_enhanced_confidence', 0):.2f}")
                            with metric_col3:
                                st.metric("Chunks Used", comparison.get('chunks_used', 0))
                                st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
                            
                            # Use the better result for chunk display
                            better_result = llm_result if llm_result.get('confidence', 0) > extractive_result.get('confidence', 0) else extractive_result
                            chunks_used = better_result.get('chunks_used', [])
                            
                        else:
                            # Single method query
                            use_llm = query_method == "LLM Enhanced Only"
                            result = rag_system.query(
                                query_input, 
                                top_k=top_k, 
                                threshold=similarity_threshold,
                                use_llm=use_llm
                            )
                            
                            # Display single result
                            st.markdown('<h3 class="section-header">📋 Query Result</h3>', unsafe_allow_html=True)
                            
                            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                            st.markdown(f"**Answer:** {result.get('answer', 'No answer generated')}")
                            st.markdown(f"**Method:** {result.get('method', 'Unknown')}")
                            st.markdown(f"**Confidence:** {result.get('confidence', 0):.2f}")
                            st.markdown(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            chunks_used = result.get('chunks_used', [])
                        
                        # Display chunks used
                        if chunks_used:
                            st.markdown(f'<h4>📄 Chunks Used for Answer Generation ({len(chunks_used)} chunks)</h4>', unsafe_allow_html=True)
                            
                            for i, chunk_info in enumerate(chunks_used):
                                if isinstance(chunk_info, dict):
                                    chunk_text = chunk_info.get('chunk_text', '')
                                    similarity_score = chunk_info.get('similarity_score', 0.0)
                                    chunk_index = chunk_info.get('chunk_index', i)
                                else:
                                    chunk_text = str(chunk_info)
                                    similarity_score = 0.0
                                    chunk_index = i
                                
                                with st.expander(f"Chunk {chunk_index + 1} (Similarity: {similarity_score:.3f})"):
                                    st.markdown('<div class="chunk-box">', unsafe_allow_html=True)
                                    st.write(chunk_text)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                    
                                    # Highlight matching terms
                                    query_terms = query_input.lower().split()
                                    highlighted_text = chunk_text
                                    for term in query_terms:
                                        if term in highlighted_text.lower():
                                            highlighted_text = highlighted_text.replace(
                                                term, f"**{term}**"
                                            )
                                    
                                    if highlighted_text != chunk_text:
                                        st.markdown("**Highlighted matches:**")
                                        st.markdown(highlighted_text)
                        else:
                            st.warning("⚠️ No relevant chunks found for this query. Try adjusting the similarity threshold or rephrasing your question.")
                        
                        # Add to query history
                        query_record = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'query': query_input,
                            'method': query_method,
                            'answer': result.get('answer', 'No answer') if query_method != "Both Methods (Comparison)" else "Comparison result",
                            'confidence': result.get('confidence', 0) if query_method != "Both Methods (Comparison)" else 0,
                            'chunks_used': len(chunks_used),
                            'processing_time': result.get('processing_time', 0)
                        }
                        st.session_state.query_history.append(query_record)
                        
                        # # Fallback message explanation
                        # st.markdown('<h4>ℹ️ Fallback Mechanism</h4>', unsafe_allow_html=True)
                        # st.info("""
                        # **Fallback Strategy:**
                        # 1. If similarity threshold is not met, the system returns a "no relevant content found" message
                        # 2. If LLM generation fails, the system automatically falls back to extractive method
                        # 3. If no chunks are retrieved, the system provides a helpful message suggesting query refinement
                        # 4. All fallbacks are logged and tracked for system improvement
                        # """)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing query: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a question")
    
    with tab3:
        # st.markdown('<h2 class="section-header">📊 Analytics Dashboard</h2>', unsafe_allow_html=True)
        
        if st.session_state.query_history:
            # Query history
            st.markdown('<h3>📈 Query History</h3>', unsafe_allow_html=True)
            
            history_df = pd.DataFrame(st.session_state.query_history)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", len(history_df))
            with col2:
                avg_confidence = history_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
            with col3:
                avg_chunks = history_df['chunks_used'].mean()
                st.metric("Avg Chunks Used", f"{avg_chunks:.1f}")
            with col4:
                avg_time = history_df['processing_time'].mean()
                st.metric("Avg Processing Time", f"{avg_time:.2f}s")
            
            # # Visualizations
            # if len(history_df) > 1:
            #     fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                
            #     # Confidence over time
            #     axes[0, 0].plot(history_df.index, history_df['confidence'], marker='o')
            #     axes[0, 0].set_title('Confidence Over Time')
            #     axes[0, 0].set_xlabel('Query Number')
            #     axes[0, 0].set_ylabel('Confidence')
                
            #     # Processing time over time
            #     axes[0, 1].plot(history_df.index, history_df['processing_time'], marker='s', color='orange')
            #     axes[0, 1].set_title('Processing Time Over Time')
            #     axes[0, 1].set_xlabel('Query Number')
            #     axes[0, 1].set_ylabel('Time (seconds)')
                
            #     # Chunks used distribution
            #     axes[1, 0].hist(history_df['chunks_used'], bins=10, alpha=0.7, color='green')
            #     axes[1, 0].set_title('Chunks Used Distribution')
            #     axes[1, 0].set_xlabel('Number of Chunks')
            #     axes[1, 0].set_ylabel('Frequency')
                
            #     # Method usage
            #     method_counts = history_df['method'].value_counts()
            #     axes[1, 1].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%')
            #     axes[1, 1].set_title('Method Usage')
                
            #     plt.tight_layout()
            #     st.pyplot(fig)
            
            # Detailed history table
            st.markdown('<h4>📋 Detailed Query History</h4>', unsafe_allow_html=True)
            st.dataframe(history_df, use_container_width=True)
            
            # Export functionality
            if st.button("📥 Export Query History"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"rag_query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No query history available. Start asking questions to see analytics!")
    
    with tab4:
        st.markdown('<h2 class="section-header">🔧 System Details</h2>', unsafe_allow_html=True)
        
        # System information
        system_info = rag_system.get_system_info()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h4>🤖 Model Information</h4>', unsafe_allow_html=True)
            st.info(f"**Embedding Model:** {system_info.get('embedding_model', 'Unknown')}")
            st.info(f"**LLM Model:** {system_info.get('llm_model', 'None')}")
            st.info(f"**LLM Available:** {'✅ Yes' if system_info.get('llm_available', False) else '❌ No'}")
            st.info(f"**Device:** {system_info.get('device', 'Unknown')}")
        
        with col2:
            st.markdown('<h4>📊 Content Status</h4>', unsafe_allow_html=True)
            st.info(f"**Content Loaded:** {'✅ Yes' if system_info.get('content_loaded', False) else '❌ No'}")
            st.info(f"**Number of Chunks:** {system_info.get('num_chunks', 0)}")
            st.info(f"**Embeddings Created:** {'✅ Yes' if system_info.get('embeddings_created', False) else '❌ No'}")
            st.info(f"**Embedding Dimensions:** {system_info.get('embedding_dimensions', 'N/A')}")
        
        # Technical details
        st.markdown('<h4>🔧 Technical Implementation</h4>', unsafe_allow_html=True)
        
        with st.expander("Vector Similarity Method Details"):
            st.markdown("""
            **Cosine Similarity Implementation:**
            
            1. **Embedding Generation**: Uses sentence-transformers to convert text chunks into dense vector representations
            2. **Query Processing**: Converts user query into the same vector space
            3. **Similarity Calculation**: Computes cosine similarity between query vector and all chunk vectors
            4. **Ranking**: Sorts chunks by similarity score in descending order
            5. **Filtering**: Applies similarity threshold to filter relevant chunks
            6. **Diversity**: Implements diversity filtering to avoid redundant chunks
            
            **Formula**: `similarity = (A · B) / (||A|| × ||B||)`
            
            **Advantages**:
            - Normalized similarity (0-1 range)
            - Handles variable document lengths
            - Computationally efficient
            - Semantic understanding through embeddings
            """)
        
        with st.expander("Chunk Extraction Strategy"):
            st.markdown("""
            **Answer Extraction Process:**
            
            1. **Keyword Matching**: Identifies overlapping terms between query and chunks
            2. **Sentence Scoring**: Scores individual sentences based on relevance
            3. **Context Preservation**: Maintains sentence boundaries for coherent answers
            4. **Length Optimization**: Balances completeness with conciseness
            5. **Fallback Handling**: Provides alternative responses when no relevant content is found
            
            **Scoring Factors**:
            - Chunk similarity score (60% weight)
            - Word overlap ratio (40% weight)
            - Sentence length consideration
            - Position within chunk
            """)
        
        with st.expander("Local LLM Integration"):
            st.markdown("""
            **LLM Enhancement Process:**
            
            1. **Context Preparation**: Combines top-K relevant chunks as context
            2. **Prompt Engineering**: Creates structured prompts for better responses
            3. **Generation Control**: Uses temperature and token limits for quality
            4. **Fallback Mechanism**: Automatically switches to extractive method on failure
            5. **Response Cleaning**: Post-processes generated text for quality
            
            **Benefits**:
            - More natural, coherent responses
            - Better context integration
            - Improved answer quality
            - No external API dependencies
            - Privacy-preserving (local processing)
            """)
        
        # Performance metrics
        if hasattr(rag_system, 'embeddings') and rag_system.embeddings is not None:
            st.markdown('<h4>📈 Performance Metrics</h4>', unsafe_allow_html=True)
            
            embedding_size = rag_system.embeddings.nbytes / (1024 * 1024)  # MB
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            with perf_col1:
                st.metric("Embedding Memory", f"{embedding_size:.2f} MB")
            with perf_col2:
                st.metric("Vector Dimensions", rag_system.embeddings.shape[1])
            with perf_col3:
                st.metric("Total Vectors", rag_system.embeddings.shape[0])
        
        # # System improvements
        # st.markdown('<h4>🚀 System Improvements</h4>', unsafe_allow_html=True)
        # improvements = system_info.get('improvements', [])
        # for improvement in improvements:
        #     st.success(f"✅ {improvement}")

else:
    st.error("❌ RAG system not initialized. Please check the sidebar to initialize the system.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>Advanced RAG System with Local LLM Integration | Built with Streamlit</p>
    <p>Features: Vector Embeddings • Cosine Similarity • Extractive QA • Local LLM • Comprehensive Analytics</p>
</div>
""", unsafe_allow_html=True)