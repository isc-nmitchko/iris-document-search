import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers.utils.import_utils import is_flash_attn_2_available
import iris
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from .MUVERA import MUVERAEncoder
import numpy as np
from pathlib import Path
import os
from hashlib import sha256
import matplotlib.pyplot as plt
import openai
import base64
from io import BytesIO

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def format_vector(vec):
    # Convert to numpy array, flatten if needed, and format to fixed-point decimals
    arr = np.array(vec).flatten()
    return (
        ",".join(f"{x}" for x in arr) + ""
    )  # 8 decimal places, no scientific notation


class IRISColPaliRAG:
    def __init__(
        self,
        iris_connection_params,
        model_name="vidore/colpali-v1.3",
        enable_encoding=True,
        enable_VLM=False,
        RAG_CUDA_DEVICE="auto",
        VLM_CUDA_DEVICE="auto",
        sql_collection_name="DocumentEmbeddings",
    ):
        # Initialize IRIS connection
        self.iris_connection_params = iris_connection_params
        self.connection = iris.connect(**iris_connection_params)
        self.cursor = self.connection.cursor()
        self.encoder = MUVERAEncoder(
            k_sim=4,  # Creates 16 clusters per repetition
            d_proj=32,  # Project to 32 dimensions per block
            R_reps=16,  # 16 repetitions: 16 * 16 * 32 = 8192 before final projection
            d_final=8192,  # Final output dimension for RAG
            fill_empty_clusters=True,
            seed=42,
        )

        self.enable_encoding = enable_encoding
        if self.enable_encoding:
            # Initialize ColPali if encoding is enabled
            self.model = ColQwen2_5.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=RAG_CUDA_DEVICE,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_2_available() else None
                ),
            ).eval()
            self.processor = ColQwen2_5_Processor.from_pretrained(
                model_name, use_fast=True
            )
        self.enable_VLM = enable_VLM
        if self.enable_VLM:
            self.VLMmodel = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype="auto",
                device_map=VLM_CUDA_DEVICE,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_2_available() else None
                ),
            )
            self.VLMprocessor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct"
            )

        self.sql_collection_name = sql_collection_name

        self.CREATE_TABLE_STATEMENT = f"""

            CREATE TABLE IF NOT EXISTS {self.sql_collection_name} (
                %PUBLICROWID,
                DocumentID VARCHAR(255),
                PageNumber INT,
                FileName VARCHAR(255),
                Embedding VECTOR(FLOAT, 8192),
                CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )

        """

        self.INSERT_SQL = f"""
        
            INSERT %NOCHECK INTO {self.sql_collection_name} (DocumentID, PageNumber, FileName, Embedding)
            VALUES (
                ?,
                ?,
                ?,
                TO_VECTOR(?, FLOAT)
            )
        
        """

        self._setup_tables()

    def change_collection(self, collection_name="DocumentEmbeddings"):
        self.sql_collection_name = collection_name

        self.CREATE_TABLE_STATEMENT = f"""

            CREATE TABLE IF NOT EXISTS {self.sql_collection_name} (
                %PUBLICROWID,
                DocumentID VARCHAR(255),
                PageNumber INT,
                FileName VARCHAR(255),
                Embedding VECTOR(FLOAT, 8192),
                CreatedDate TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )

        """

        self.INSERT_SQL = f"""
        
            INSERT %NOCHECK INTO {self.sql_collection_name} (DocumentID, PageNumber, FileName, Embedding)
            VALUES (
                ?,
                ?,
                ?,
                TO_VECTOR(?, FLOAT)
            )
        
        """
        self._setup_tables()

    def _setup_tables(self):
        """Create necessary tables and indexes"""
        self.cursor.execute(self.CREATE_TABLE_STATEMENT)
        try:
            self.cursor.execute(
                f"""
                CREATE INDEX VDPANN 
                    ON {self.sql_collection_name} (Embedding)
                    AS HNSW(Distance='dotproduct')
                    DEFER
            """
            )
        except iris.dbapi.ProgrammingError:
            print("Error creating HNSW Index, okay but performance will suffer")
        self.connection.commit()

    def _reconnect(self):
        self.connection = iris.connect(**self.iris_connection_params)
        self.cursor = self.connection.cursor()

    def hash_pdf(self, pdf_path):
        """Generate a SHA-256 hash of the PDF to use as a document ID"""
        hasher = sha256()
        with open(pdf_path, "rb") as pdf_file:
            # Read the file in chunks to handle large files
            for chunk in iter(lambda: pdf_file.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def encode_image_to_base64(self, image_path):
        """
        Encode image to base64 string for OpenAI API

        Args:
            image_path: Path to image file or PIL Image object

        Returns:
            str: Base64 encoded image
        """
        if isinstance(image_path, str):
            # If it's a file path
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(image_path, Image.Image):
            # If it's a PIL Image object
            buffered = BytesIO()
            image_path.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise ValueError(
                "image_path must be a file path string or PIL Image object"
            )

    def call_openai_vlm_stream(
        self,
        images,
        query,
        api_key=None,
        base_url="http://ai-server-1.local/internvlm/v1",
        model="salvator",
    ):
        """
        Call local OpenAI-compatible API for VLM inference with streaming response
        Args:
            images: List of image paths or PIL Image objects
            query: User query string
            api_key: API key (can be dummy for local APIs)
            base_url: Base URL for local OpenAI-compatible API
            model: Model name to use
        Yields:
            str: Streaming chunks of generated answer from VLM
        """
        system_prompt = "You are an expert professional PDF analyst who gives rigorous in-depth answers."
        # Set up OpenAI client for local API
        client = openai.OpenAI(
            api_key=api_key or "dummy-key",  # Many local APIs don't require real keys
            base_url=base_url,
        )

        # Prepare messages with images
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Based on the following images, please answer this question: {query}",
                    }
                ],
            },
        ]

        # Add images to the message
        for image in images[:5]:  # Limit to top 5 images to avoid token limits
            try:
                base64_image = self.encode_image_to_base64(image)
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
            except Exception as e:
                print(f"Warning: Could not encode image {image}: {e}")
                continue

        # Make streaming API call
        try:
            stream = client.chat.completions.create(
                model=model,  # Use your local model name
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
                stream=True,  # Enable streaming
            )

            # Yield chunks as they arrive
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"Error calling VLM API: {str(e)}"

    def call_openai_vlm(
        self,
        images,
        query,
        api_key=None,
        base_url="http://ai-server-1.local/internvlm/v1",
        model="salvator",
    ):
        """
        Call local OpenAI-compatible API for VLM inference

        Args:
            images: List of image paths or PIL Image objects
            query: User query string
            api_key: API key (can be dummy for local APIs)
            base_url: Base URL for local OpenAI-compatible API

        Returns:
            str: Generated answer from VLM
        """
        # Set up OpenAI client for local API
        client = openai.OpenAI(
            api_key=api_key or "dummy-key",  # Many local APIs don't require real keys
            base_url=base_url,
        )
        system_prompt = "You are an expert professional PDF analyst who gives rigorous in-depth answers."
        # Prepare messages with images
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Based on the following images, please answer this question: {query}",
                    }
                ],
            },
        ]

        # Add images to the message
        for image in images[:5]:  # Limit to top 5 images to avoid token limits
            try:
                base64_image = self.encode_image_to_base64(image)
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
            except Exception as e:
                print(f"Warning: Could not encode image {image}: {e}")
                continue

        # Make API call
        try:
            response = client.chat.completions.create(
                model=model,  # Use your local model name
                messages=messages,
                max_tokens=2048,
                temperature=0.7,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error calling VLM API: {str(e)}"

    def generate_answer(self, images, query):
        """
        Generate an answer using Vision Language Model

        Args:
            images: List of top reranked images
            query: Original user query

        Returns:
            str: Generated answer from VLM
        """
        return self.call_openai_vlm(query=query, images=images)

    def process_document_page(self, image_path_or_pil):
        """Process a single document page and return MUVERA FDE"""
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil

        batch_images = self.processor.process_images([image]).to(self.model.device)

        with torch.no_grad():
            # Get multi-vector embeddings from ColPali
            multi_vector_embedding = self.model.forward(**batch_images)

            # Convert to numpy for MUVERA processing
            # multi_vecs = multi_vector_embedding.cpu().numpy()[0]
            multi_vecs = multi_vector_embedding.to(torch.float32).cpu().numpy()[0]

            # Apply MUVERA FDE encoding
            # single_vector_embedding = muvera_fde_encode(multi_vecs)
            single_vector_embedding = self.encoder.encode_document(multi_vecs)

        return single_vector_embedding

    def ingest_pdf(self, path):
        """Process and store PDF embeddings"""
        pdf_paths = []

        if os.path.isfile(path):
            pdf_paths.append(path)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        pdf_paths.append(os.path.join(root, file))

        for pdf_path in pdf_paths:
            document_id = self.hash_pdf(pdf_path)
            page_images = convert_from_path(pdf_path=pdf_path, dpi=400)
            for page_num, image in enumerate(page_images, 1):
                fde_embedding = self.process_document_page(image)
                embedding = format_vector(fde_embedding)
                self.cursor.execute(
                    self.INSERT_SQL, [document_id, page_num, pdf_path, embedding]
                )
                self.connection.commit()
        # self.cursor.execute(
        #     f"""
        #     BUILD INDEX FOR TABLE {self.sql_collection_name}
        #     """
        # )

    @staticmethod
    def show_results(results, images=None, show=True):
        # Filter the results to get the specific images/pages required

        images_to_display = []
        if images is not None:
            images_to_display = images
        else:
            for page_number, file_path, _ in results:
                # Convert only the specific page from the PDF to an image
                images = convert_from_path(
                    file_path, first_page=page_number, last_page=page_number
                )
                if images:
                    images_to_display.extend(images)

        if show:
            # Display all the collected images as subplots
            num_images = len(images_to_display)
            num_rows = num_images // 5 + (1 if num_images % 5 > 0 else 0)
            fig, axes = plt.subplots(num_rows, 5, figsize=(20, 4 * num_rows))
            axes = axes.flatten()

            for i, img in enumerate(images_to_display):
                if i < len(axes):
                    ax = axes[i]
                    ax.imshow(img)
                    # ax.set_title(f"Page {results[i][0]}")
                    ax.axis("off")

            # Hide any unused subplot axes
            for j in range(num_images, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.show()
        return images_to_display

    def search(self, query, top_k=10):
        """Search documents using MUVERA-encoded embeddings"""
        # Process query through ColPali -> MUVERA pipeline
        batch_queries = self.processor.process_queries([query]).to(self.model.device)

        with torch.no_grad():
            multi_vector_query = self.model.forward(**batch_queries)
            query_multi_vecs = multi_vector_query.to(torch.float32).cpu().numpy()[0]
            # query_fde = muvera_fde_encode(query_multi_vecs)
            query_fde = self.encoder.encode_query(query_multi_vecs)

        # Convert to string for IRIS
        query_vector_str = format_vector(query_fde)

        # Perform vector similarity search
        search_sql = f"""
        SELECT TOP {top_k}
            PageNumber,
            FileName,
            COUNT(*)
        
        FROM {self.sql_collection_name}
        
        ORDER BY VECTOR_DOT_PRODUCT(Embedding,TO_VECTOR(?, FLOAT)) 
            DESC
        """

        # 	Fixed this error when connection is dropped over a long period of time
        #       self.cursor.execute(search_sql, [query_vector_str])

        try:
            self.cursor.execute(search_sql, [query_vector_str])
        except iris.dbapi.ProgrammingError as e:
            # Check for connection closed / SSL error in the exception text
            if "SSL_ERROR_ZERO_RETURN" in str(e) or "connection closed" in str(e):
                # Attempt reconnect
                self._reconnect()
                # Re-try the query after reconnecting
                self.cursor.execute(search_sql, [query_vector_str])
            else:
                # Not a connection issue, re-raise
                raise

        results = self.cursor.fetchall()
        return results

    def get_statistics(self):
        search_sql = f"""
        SELECT TOP 1 count(*), COUNT(DISTINCT(DocumentID))
        FROM {self.sql_collection_name}
        """

        try:
            self.cursor.execute(search_sql)
        except iris.dbapi.ProgrammingError as e:
            # Check for connection closed / SSL error in the exception text
            if "SSL_ERROR_ZERO_RETURN" in str(e) or "connection closed" in str(e):
                # Attempt reconnect
                self._reconnect()
                # Re-try the query after reconnecting
                self.cursor.execute(search_sql)
            else:
                # Not a connection issue, re-raise
                raise

        results = self.cursor.fetchall()
        return results

    def rerank(self, images, query):
        batch_queries = self.processor.process_queries([query]).to(self.model.device)
        batch_images = self.processor.process_images(images).to(self.model.device)

        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
            query_embeddings = self.model(**batch_queries)

        # Reorder the images
        similarity_scores = self.processor.score_multi_vector(
            query_embeddings, image_embeddings
        ).numpy()[
            0
        ]  # Convert to numpy array if it isn't already

        return similarity_scores

    def close(self):
        """Clean up connections"""
        self.cursor.close()
        self.connection.close()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context related to this object."""
        self.close()
