# %%
import iris
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
import time
import threading
from rich.console import Console
from rich.spinner import Spinner
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text
from getpass import getpass
# import gc
import os
from irisutils import (
    get_env_variables,
    IRISColPaliRAG as IRISDocCollection,
    get_iris_connection_settings,
)
import argparse

try:
    irissettings = get_env_variables()
except:
    irissettings = []
args = None


def search_repl(rag: IRISDocCollection, top_k=5, inference_k=3, show=False):
    """
    Interactive REPL shell for RAG search functionality with beautiful loading animations.

    Args:
        rag: Your RAG object with search, show_results, and rerank methods

    Required installation:
        pip install rich
    """
    console = Console()

    console.print("Document QA", style="bold blue")
    console.print("================", style="blue")
    console.print(
        "Enter your search queries below. Type 'quit', 'exit', or 'q' to stop."
    )
    console.print()

    while True:
        try:
            # Get user input
            query = console.input("[bold green]Question>[/] ").strip()

            # Check for quit commands
            if query.lower() in ["quit", "exit", "q", ""]:
                console.print("Goodbye!", style="bold yellow")
                break

            # Start timing the entire query
            query_start_time = time.time()

            console.print(f"Searching for: [bold cyan]'{query}'[/]")
            console.print("-" * 40)

            # Search with loading animation
            with Live(
                Spinner("dots", text="🔍 Searching database..."), refresh_per_second=10
            ) as live:
                search_start = time.time()
                results = rag.search(query, top_k=top_k)
                search_time = time.time() - search_start
                live.update(Spinner("dots", text="📋 Gathering results..."))
                images = rag.show_results(results, show=False)

            console.print(f"✅ Search completed in [bold green]{search_time:.2f}s[/]")
            console.print(f"🔍 Search Scanned [bold green]{results[0][2]} Documents[/]")

            # Rerank with loading animation
            with Live(
                Spinner("bouncingBall", text="🧠 Reranking results with AI..."),
                refresh_per_second=10,
            ) as live:
                rerank_start = time.time()
                rerank_scores = rag.rerank(images, query)
                rerank_time = time.time() - rerank_start

            console.print(
                f"✅ Reranking completed in [bold green]{rerank_time:.2f}s[/]"
            )

            # Sort by similarity (highest first)
            with console.status("📊 Sorting results..."):
                sorted_indices = np.argsort(rerank_scores)[::-1]
                reranked_images = [images[i] for i in sorted_indices]
                sorted_scores = rerank_scores[sorted_indices]

            # Show reranked results
            # rag.show_results(None, reranked_images, show=show)

            console.print("\n[bold magenta]🔮 AI Answer:[/]")
            console.print("=" * 50)
            # Generate answer with VLM using top reranked images
            # with Live(
            #     Spinner(
            #         "arc", text="🤖 Generating answer with Vision Language Model..."
            #     ),
            #     refresh_per_second=10,
            # ) as live:
            #     vlm_start = time.time()
            #     # Use top N images for VLM (e.g., top 3-5 most relevant)
            #     top_images = reranked_images[:inference_k]  # Adjust number as needed
            #     vlm_answer = rag.generate_answer(top_images, query)
            #     vlm_time = time.time() - vlm_start
                
            accumulated_text = ""
    
            # Method 2: Using Live display (more efficient, no flicker)
            with Live(console=console, refresh_per_second=10) as live:
                vlm_start = time.time()
                top_images = reranked_images[:inference_k]
                for chunk in rag.call_openai_vlm_stream(top_images, query):
                    accumulated_text += chunk
                    md = Markdown(accumulated_text)
                    live.update(md)
                    time.sleep(0.05)
                vlm_time = time.time() - vlm_start


            # Display the VLM answer
            # console.print(f"[white]{vlm_answer}[/]")
            console.print("=" * 50)
            console.print(f"✅ VLM answer generated in [bold green]{vlm_time:.2f}s[/]")
            # Calculate and display timing summary
            total_time = time.time() - query_start_time

            console.print("\n[bold blue]📈 Performance Summary:[/]")
            console.print(f"   Search:   {search_time:.2f}s")
            console.print(f"   Rerank:   {rerank_time:.2f}s")
            console.print(f"   Total:    {total_time:.2f}s")
            
            if show:
                rag.show_results(None, top_images, show=show)

            console.print("\nPress [ENTER] to continue searching...", style="dim")
            input()
            console.print()

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Goodbye![/]")
            break
        except Exception as e:
            console.print(f"[bold red]Error occurred:[/] {e}")
            console.print("Please try again.")
            console.print()

def setup():
    host = input("IRIS SQL Endpont or SuperServer>")
    port = input("IRIS SQL Port>")
    env = f"""
# IRIS Host
SQL_HOSTNAME={host}
# IRIS Port
SQL_PORT={port}
# IRIS Namespace
SQL_NAMESPACE={input("IRIS SQL Namespace>")}
# IRIS USER
SQL_USERNAME={input("IRIS SQL User>")}
# IRIS Password, leave empty for getpass
SQL_PASSWORD={getpass("IRIS SQL Password>")}
# SSL configuration
SQL_SSLCONFIG=SimpleSSLConfig
# Connection timeout
SQL_TIMEOUT=100
# False for remote connections
SQL_SHAREDMEMORY=False
# Necessary, for windows or sql connections
# ISC_SSLconfigurations="./connection/SSLDefs.ini"

# LOCAL AI device for creating embeddings
RAG_DEVICE_MAP="cuda:0"
VLM_DEVICE_MAP="cuda:0"
# MODEL SLUG compatable with colpali like models
# conomic like models
# also, you can try 3b models
MODEL_SLUG="nomic-ai/colnomic-embed-multimodal-7b"
# Change for your system
CUDA_VISIBLE_DEVICES=0
# OpenAI API Endpoint
OPENAI_API_BASE_URL="https://api.openai.com/v1/"
# OpenAI API Key
OPENAI_API_KEY={getpass("OPENAI_API_KEY (leave blank to edit .env later)>")}

OPENAI_API_MODEL="gpt-5"
    """
    SSLDefs = f"""
[SimpleSSLConfig]
CAFile=
CertFile=./connection/connection.pem
KeyFile=
Password=
KeyType=2 
Protocols=28 
CipherList=ALL:!aNULL:!eNULL:!EXP:!SSLv2 
VerifyPeer=0 
VerifyDepth=9 
TLSMinVersion=16
TLSMaxVersion=32
Ciphersuites=TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256

[MyServer] 
Address={host}
Port={port}
SSLConfig=SimpleSSLConfig"""
    with open("./connection/SSLDefs.ini", "w+") as ssldefs:
        ssldefs.write(SSLDefs)
    with open(".env-new", "w+") as envfile:
        envfile.write(env)
        print("="*50)
        print("Wrute env file to '.env-new', review and copy to '.env'")
        print("Copy your IRIS cloud SQL connection certificate to ./connection/connection.pem")
        print("Then re-run script or use streamlit run ui2.py")
    exit()


def main():
    parser = argparse.ArgumentParser(
        description="Script for running IRIS RAG using advanced document retrieval"
    )
    parser.add_argument('--setup', action='store_true', help="Setup environment variables")
    parser.add_argument(
        "--embed",
        type=str,
        help="Calculates and stores VLM Embededding data from the path or directory.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Enter the search interface against the repository",
    )
    parser.add_argument(
        "--topk", type=int, help="Number of results to return from vector search", default=10
    )
    
    parser.add_argument(
        "--inference-k", type=int, help="Number of images to send to VLM for inference, must be less than topk", default=3
    )

    parser.add_argument(
        "--show",
        help="Show intermediate results visually, must manually close the windows to continue to the next step.",
        action="store_true",
    )
    
    parser.add_argument(
        "--localVLM",
        help="Spins up a Qwen2.5 VLM locally. Requires enough GPU VRAM to host locally. Doesn't try quantization",
        action="store_true",
    )

    args = parser.parse_args()


    if args.setup:
        setup()
    iris_conn = get_iris_connection_settings(irissettings)
    with IRISDocCollection(
        iris_connection_params=iris_conn,
        model_name=irissettings["MODEL_SLUG"],
        VLM_CUDA_DEVICE=irissettings["VLM_DEVICE_MAP"],
        RAG_CUDA_DEVICE=irissettings["RAG_DEVICE_MAP"],
        enable_VLM=args.localVLM
    ) as rag:
        if args.embed:
            rag.ingest_pdf(
                args.embed,
            )
        if args.search:
            search_repl(rag, args.topk, args.inference_k, args.show)


if __name__ == "__main__":
    main()
