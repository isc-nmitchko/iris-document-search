import streamlit as st
import iris
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
import threading
from PIL import Image
import io
import base64

# import gc
import os
from irisutils import (
    get_env_variables,
    IRISColPaliRAG as IRISDocCollection,
    get_iris_connection_settings,
)
import argparse

# Page config
st.set_page_config(
    page_title="Document QA System",
    page_icon="🔍",
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
    
    .search-box {
        margin: 2rem 0;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .answer-section {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .stSpinner > div > div {
        border-top-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system with caching"""
    try:
        irissettings = get_env_variables()
        iris_conn = get_iris_connection_settings(irissettings)
        
        rag = IRISDocCollection(
            iris_connection_params=iris_conn,
            model_name=irissettings["MODEL_SLUG"],
            VLM_CUDA_DEVICE=irissettings["VLM_DEVICE_MAP"],
            RAG_CUDA_DEVICE=irissettings["RAG_DEVICE_MAP"],
            enable_VLM=st.session_state.get('enable_local_vlm', False)
        )
        return rag, irissettings
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None, None

def display_images_grid(images, scores=None, max_cols=3):
    """Display images in a grid layout"""
    if not images:
        return
    
    cols = st.columns(min(len(images), max_cols))
    
    for i, image_item in enumerate(images):
        col_idx = i % max_cols
        with cols[col_idx]:
            try:
                # Handle different image types
                if isinstance(image_item, str):
                    # String path - try to load from file
                    if os.path.exists(image_item):
                        image = Image.open(image_item)
                        st.image(image, caption=f"Result {i+1}", use_container_width=True)
                    else:
                        st.warning(f"Image not found: {image_item}")
                elif hasattr(image_item, 'save') or hasattr(image_item, 'show'):
                    # PIL Image object or similar
                    st.image(image_item, caption=f"Result {i+1}", use_container_width=True)
                elif hasattr(image_item, 'path') and hasattr(image_item.path, 'as_posix'):
                    # PosixPath or similar path object
                    image_path = str(image_item.path)
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption=f"Result {i+1}", use_container_width=True)
                    else:
                        st.warning(f"Image not found: {image_path}")
                elif hasattr(image_item, '__fspath__'):
                    # Path-like object
                    image_path = os.fspath(image_item)
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption=f"Result {i+1}", use_container_width=True)
                    else:
                        st.warning(f"Image not found: {image_path}")
                else:
                    # Try to convert to string and use as path
                    image_path = str(image_item)
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        st.image(image, caption=f"Result {i+1}", use_container_width=True)
                    else:
                        # Last resort - pass directly to streamlit and let it handle
                        st.image(image_item, caption=f"Result {i+1}", use_container_width=True)
                
                # Add relevance score if available
                if scores is not None and i < len(scores):
                    st.caption(f"Relevance Score: {scores[i]:.3f}")
                    
            except Exception as e:
                st.error(f"Error loading image {i+1}: {e}")
                # Debug information
                st.write(f"Image type: {type(image_item)}")
                st.write(f"Image value: {repr(image_item)}")
                
                # Try to display some attributes for debugging
                if hasattr(image_item, '__dict__'):
                    st.write(f"Attributes: {list(image_item.__dict__.keys())}")

def stream_markdown_response(rag, images, query):
    """Stream the VLM response and display it in markdown"""
    # Create a placeholder for the streaming response
    response_placeholder = st.empty()
    accumulated_text = ""
    
    try:
        for chunk in rag.call_openai_vlm_stream(images, query):
            accumulated_text += chunk
            # Update the placeholder with accumulated text
            response_placeholder.markdown(accumulated_text)
            time.sleep(0.05)  # Small delay for visual streaming effect
    except Exception as e:
        st.error(f"Error during VLM streaming: {e}")
        return ""
    
    return accumulated_text

def main():
    # Header
    st.markdown('<h1 class="main-header">IRIS Colnomic Doc QA</h1>', unsafe_allow_html=True)
    # st.markdown("Ask questions about your documents and get AI-powered answers with visual evidence.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Search parameters
        top_k = st.slider("Search Results (top_k)", min_value=1, max_value=50, value=7, 
                         help="Number of results to return from vector search")
        
        inference_k = st.slider("Images for Inference", min_value=1, max_value=min(5, top_k), value=3,
                               help="Number of images to send to VLM for inference")
        
        show_intermediate = st.checkbox("Show Intermediate Results", value=True,
                                      help="Display search results visually")
        
        # Search mode selection
        search_mode = st.radio(
            "Search Mode",
            options=["Full Q&A", "Images Only"],
            index=0,
            help="Choose whether to generate AI answers or just show matching images"
        )
        
        enable_local_vlm = st.checkbox("Enable Local VLM", value=False,
                                     help="Use local Qwen2.5 VLM (requires GPU)")
        
        # Store in session state
        st.session_state['enable_local_vlm'] = enable_local_vlm
        
        st.divider()
        
        # Document ingestion section
        st.header("📄 Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents", 
            type=['pdf'], 
            accept_multiple_files=True,
            help="Upload PDF files to add to the knowledge base"
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing uploaded documents..."):
                # Here you would implement document processing
                # This is a placeholder for the actual implementation
                st.success(f"Successfully processed {len(uploaded_files)} documents!")
    
    # Initialize RAG system
    if 'rag' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            rag, irissettings = initialize_rag()
            if rag:
                st.session_state['rag'] = rag
                st.session_state['irissettings'] = irissettings
                st.success("✅ RAG system initialized successfully!")
            else:
                st.error("❌ Failed to initialize RAG system")
                st.stop()
    else:
        rag = st.session_state['rag']
    
    # Main search interface
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    
    # Search query input
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="Enter your question here...",
        help="Type your question and press Enter to search"
    )
    
    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process search when button is clicked or query is entered
    if (search_button) and query.strip():
        # Initialize session state for this search
        search_key = f"search_{hash(query)}"
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Search
            status_text.text("🔍 Searching database...")
            progress_bar.progress(20)
            
            search_start = time.time()
            results = rag.search(query, top_k=top_k)
            search_time = time.time() - search_start
            
            progress_bar.progress(40)
            status_text.text("📋 Gathering results...")
            
            raw_images = rag.show_results(results, show=False)
            
            # Convert images to a format Streamlit can handle
            images = []
            for img in raw_images:
                if hasattr(img, 'path'):
                    # If it's a custom object with a path attribute
                    images.append(str(img.path))
                elif hasattr(img, '__fspath__'):
                    # If it's a path-like object
                    images.append(os.fspath(img))
                else:
                    # Keep as-is and let display function handle it
                    images.append(img)
            
            # Step 2: Rerank
            status_text.text("🧠 Reranking results with AI...")
            progress_bar.progress(60)
            
            rerank_start = time.time()
            rerank_scores = rag.rerank(images, query)
            rerank_time = time.time() - rerank_start
            
            # Sort by similarity
            sorted_indices = np.argsort(rerank_scores)[::-1]
            reranked_images = [images[i] for i in sorted_indices]
            sorted_scores = rerank_scores[sorted_indices]
            
            progress_bar.progress(80)
            status_text.text("🤖 Generating AI answer...")
            
            progress_bar.progress(80)
            
            # Check if we should skip VLM and just show images
            if search_mode == "Images Only":
                status_text.text("📸 Preparing image results...")
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                # Display results without VLM processing
                # st.success("✅ Image search completed successfully!")
                
                # Performance metrics
                total_time = search_time + rerank_time
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Search Time", f"{search_time:.2f}s")
                with col2:
                    st.metric("Rerank Time", f"{rerank_time:.2f}s")
                with col3:
                    st.metric("Documents Scanned", f"{results[0][2] if results else 0}")
                
                # Show image results
                st.markdown("---")
                st.markdown("## Search Results")
                st.markdown(f"Found {len(reranked_images)} relevant images for your query.")
                
                # Always show images for image-only mode
                display_images_grid(reranked_images, sorted_scores, max_cols=4)
                
            else:
                # Full Q&A mode - continue with VLM processing
                status_text.text("🤖 Generating AI answer...")
                
                # Step 3: Generate VLM answer
                vlm_start = time.time()
                top_images = reranked_images[:inference_k]
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                # Display results
                # st.success("✅ Search completed successfully!")
                
                # Performance metrics
                total_time = search_time + rerank_time
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Search Time", f"{search_time:.2f}s")
                with col2:
                    st.metric("Rerank Time", f"{rerank_time:.2f}s")
                with col3:
                    st.metric("Documents Scanned", f"{results[0][2] if results else 0}")
                with col4:
                    st.metric("Total Time", f"{total_time:.2f}s")
                
                # AI Answer Section
                st.markdown("---")
                st.markdown("## 🔮 AI Answer")
                
                # Stream the response
                vlm_answer = stream_markdown_response(rag, top_images, query)
                vlm_time = time.time() - vlm_start
                
                st.metric("🤖 VLM Response Time", f"{vlm_time:.2f}s")
                
                # Show intermediate results if enabled
                if show_intermediate and top_images:
                    st.markdown("---")
                    st.markdown("## Supporting Evidence")
                    st.markdown("These are the most relevant images used to generate the answer:")
                    
                    display_images_grid(top_images, sorted_scores[:inference_k])
                    
                    # Show all results in an expander
                    with st.expander("🔍 View All Search Results"):
                        display_images_grid(reranked_images, sorted_scores, max_cols=4)
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ An error occurred during search: {e}")
            st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <small>Document QA System powered by IRIS Vector Search and Vision Language Models</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()