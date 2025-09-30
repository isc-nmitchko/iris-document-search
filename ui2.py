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
import os
from irisutils import (
    get_env_variables,
    IRISColPaliRAG as IRISDocCollection,
    get_iris_connection_settings,
)
import argparse

# Page config
st.set_page_config(
    page_title="IRIS Document QA System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styled after Morphik.ai
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }

    .stApp > header {
        background-color: transparent;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0);
        border-radius: 15px;
        padding: 1rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f1f5f9);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }

    .search-container {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.05);
    }

    .answer-section {
        background: linear-gradient(145deg, #f0f9ff, #e0f2fe);
        border-left: 4px solid #0ea5e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .stats-container {
        background: linear-gradient(145deg, #fefce8, #fef3c7);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .upload-section {
        background: linear-gradient(145deg, #f0fdf4, #dcfce7);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }

    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.2);
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 0.75rem 1rem;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin: 2rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .progress-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag(collection_name="DocumentEmbeddings"):
    """Initialize the RAG system with caching"""
    try:
        irissettings = get_env_variables()
        iris_conn = get_iris_connection_settings(irissettings)

        rag = IRISDocCollection(
            iris_connection_params=iris_conn,
            model_name=irissettings["MODEL_SLUG"],
            VLM_CUDA_DEVICE=irissettings["VLM_DEVICE_MAP"],
            RAG_CUDA_DEVICE=irissettings["RAG_DEVICE_MAP"],
            enable_VLM=st.session_state.get('enable_local_vlm', False),
            sql_collection_name=collection_name
        )
        return rag, irissettings
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None, None

def display_images_grid(images, scores=None, max_cols=3):
    """Display images in a grid layout with modern styling"""
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
                    st.caption(f"📊 Relevance Score: {scores[i]:.3f}")

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
        for chunk in rag.call_openai_vlm_stream(images, query, os.environ.get('OPENAI_API_KEY', None),
                            os.environ.get('OPENAI_API_BASE_URL', 'http://ai-server-1.local/internvlm/v1'),
                            os.environ.get('OPENAI_API_MODEL', 'salvator')):
            accumulated_text += chunk
            # Update the placeholder with accumulated text
            response_placeholder.markdown(accumulated_text)
            time.sleep(0.05)  # Small delay for visual streaming effect
    except Exception as e:
        st.error(f"Error during VLM streaming: {e}")
        return ""

    return accumulated_text

def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    # Create data directory if it doesn't exist
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Save file
    file_path = os.path.join(data_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def main():
    # Main container
    # st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Header
    st.markdown('<h1 class="main-header">IRIS Document QA</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Intelligent document search powered by advanced AI vision models</p>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        collection_name = st.text_input(
            "Collection Name",
            value=st.session_state.get('collection_name', 'DocumentEmbeddings'),
            help="Name of the IRIS collection to use, Enter a new name to create one"
        )

        if st.button("Change Collection"):
            st.session_state['collection_name'] = collection_name
            # Clear cached RAG instance to force reinitialize with new collection
            if 'rag' in st.session_state:
                st.session_state['rag'].change_collection(collection_name)
                # del st.session_state['rag']
            st.success(f"Collection changed to: {collection_name}")
            st.rerun()

        st.divider()

        # Search parameters
        st.markdown("###  Search Settings")
        top_k = st.slider("Search Results (top_k)", min_value=1, max_value=10, value=7,
                         help="Number of results to return from vector search")

        inference_k = st.slider("Images for Inference", min_value=1, max_value=min(5, top_k), value=3,
                               help="Number of images to send to VLM for inference")

        show_intermediate = st.checkbox("Show Intermediate Results", value=True,
                                      help="Display search results visually")

        run_rerank = st.checkbox("Run Reranking?", value=True,
                                      help="Runs AI reranking after vector search for improved accuracy")

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
        st.markdown("### Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to add to the knowledge base"
        )

        if uploaded_files and st.button("📥 Process Documents"):
            with st.spinner("Processing uploaded documents..."):
                processed_count = 0
                error_count = 0

                progress_bar = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))

                        # Save file to data directory
                        file_path = save_uploaded_file(uploaded_file)

                        # Ingest PDF using RAG system
                        if 'rag' in st.session_state:
                            rag = st.session_state['rag']
                            rag.ingest_pdf(file_path)
                            processed_count += 1
                        else:
                            st.warning("RAG system not initialized. Please refresh and try again.")
                            break

                    except Exception as e:
                        error_count += 1
                        st.error(f"Error processing {uploaded_file.name}: {e}")

                progress_bar.empty()

                if processed_count > 0:
                    st.success(f"Successfully processed {processed_count} documents!")
                if error_count > 0:
                    st.warning(f"Failed to process {error_count} documents.")

    # Initialize RAG system
    current_collection = st.session_state.get('collection_name', 'DocumentEmbeddings')

    if 'rag' not in st.session_state:
        with st.spinner("🚀 Initializing RAG system..."):
            rag, irissettings = initialize_rag(current_collection)
            if rag:
                st.session_state['rag'] = rag
                st.session_state['irissettings'] = irissettings
                # st.success("✅ RAG system initialized successfully!")
            else:
                st.error("❌ Failed to initialize RAG system")
                st.stop()
    else:
        rag = st.session_state['rag']

    # Collection Statistics
    # st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    # st.markdown("### 📊 Collection Statistics")

    try:
        stats = rag.get_statistics()
        if stats and len(stats) > 0:
            total_documents = stats[0][0] if stats[0][0] else 0
            unique_documents = stats[0][1] if stats[0][1] else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Embeddings", f"{total_documents:,}")
            with col2:
                st.metric("Unique Documents", f"{unique_documents:,}")
            with col3:
                avg_embeddings = total_documents / unique_documents if unique_documents > 0 else 0
                st.metric("Avg Embeddings/Doc", f"{avg_embeddings:.1f}")
        else:
            st.info("No documents found in collection.")
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
        del st.session_state['rag']
        with st.spinner("🚀 Initializing RAG system..."):
            rag, irissettings = initialize_rag(current_collection)
            if rag:
                st.session_state['rag'] = rag
                st.session_state['irissettings'] = irissettings
                # st.success("✅ RAG system initialized successfully!")
            else:
                st.error("❌ Failed to initialize RAG system")
                st.stop()


    # st.markdown('</div>', unsafe_allow_html=True)

    # Main search interface
    # st.markdown('<div class="search-container">', unsafe_allow_html=True)

    # Search query input
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="What would you like to know about your documents?",
        help="Type your question and press Enter or click Search"
    )

    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)

    # st.markdown('</div>', unsafe_allow_html=True)

    # Process search when button is clicked or query is entered
    if (search_button) and query.strip():
        # Initialize session state for this search
        search_key = f"search_{hash(query)}"

        # Progress tracking
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Search
            status_text.text("Searching database...")
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
            # status_text.text("🧠 Reranking results with AI...")
            # progress_bar.progress(60)

            # rerank_start = time.time()
            # if run_rerank:
            #     rerank_scores = rag.rerank(images, query)
            # else:
            #     rerank_scores = [i for i in range(0, len(images))]
            # rerank_time = time.time() - rerank_start

            # # Sort by similarity
            # sorted_indices = np.argsort(rerank_scores)[::-1]
            # # print(sorted_indices)
            # reranked_images = [images[i] for i in sorted_indices]
            # sorted_scores = rerank_scores[sorted_indices]
            if run_rerank:
                status_text.text("🧠 Reranking results with AI...")
                progress_bar.progress(60)

                rerank_start = time.time()
                rerank_scores = rag.rerank(images, query)
                rerank_time = time.time() - rerank_start

                # Sort by similarity (higher scores first)
                sorted_indices = np.argsort(rerank_scores)[::-1]
                reranked_images = [images[i] for i in sorted_indices]
                sorted_scores = rerank_scores[sorted_indices]
            else:
                status_text.text("📋 Using original search order...")
                progress_bar.progress(60)

                rerank_start = time.time()
                # Use original order - create numpy array for consistency
                rerank_scores = np.arange(len(images), 0, -1, dtype=float)  # Descending scores
                rerank_time = time.time() - rerank_start

                # No reordering needed
                reranked_images = images.copy()
                sorted_scores = rerank_scores.copy()


            progress_bar.progress(80)

            # Check if we should skip VLM and just show images
            if search_mode == "Images Only":
                status_text.text("📸 Preparing image results...")
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                # st.markdown('</div>', unsafe_allow_html=True)

                # Performance metrics
                total_time = search_time + rerank_time

                # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⚡ Search Time", f"{search_time:.2f}s")
                with col2:
                    st.metric("🧠 Rerank Time", f"{rerank_time:.2f}s")
                with col3:
                    st.metric("📄 Documents Scanned", f"{results[0][2] if results else 0}")
                # st.markdown('</div>', unsafe_allow_html=True)

                # Show image results
                st.markdown('<h2 class="section-header">🖼️ Search Results</h2>', unsafe_allow_html=True)
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
                # st.markdown('</div>', unsafe_allow_html=True)

                # Performance metrics
                total_time = search_time + rerank_time

                # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("⚡ Search Time", f"{search_time:.2f}s")
                with col2:
                    st.metric("🧠 Rerank Time", f"{rerank_time:.2f}s")
                with col3:
                    st.metric("📄 Documents Scanned", f"{results[0][2] if results else 0}")
                with col4:
                    st.metric("⏱️ Total Time", f"{total_time:.2f}s")
                # st.markdown('</div>', unsafe_allow_html=True)

                # AI Answer Section
                # st.markdown('<div class="answer-section">', unsafe_allow_html=True)
                st.markdown('<h2 class="section-header">AI Answer</h2>', unsafe_allow_html=True)

                # Stream the response
                vlm_answer = stream_markdown_response(rag, top_images, query)
                vlm_time = time.time() - vlm_start

                st.metric("🤖 VLM Response Time", f"{vlm_time:.2f}s")
                # st.markdown('</div>', unsafe_allow_html=True)

                # Show intermediate results if enabled
                if show_intermediate and top_images:
                    st.markdown('<h2 class="section-header">🔍 Supporting Evidence</h2>', unsafe_allow_html=True)
                    st.markdown("These are the most relevant images used to generate the answer:")

                    display_images_grid(top_images, sorted_scores[:inference_k])

                    # Show all results in an expander
                    with st.expander("🔍 View All Search Results"):
                        display_images_grid(reranked_images, sorted_scores, max_cols=4)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            st.error(f"❌ An error occurred during search: {e}")
            st.exception(e)

    # Footer
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown("**IRIS Document QA System** - Powered by advanced vision language models and vector search")
    # st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown('</div>', unsafe_allow_html=True)  # Close main container

if __name__ == "__main__":
    main()
