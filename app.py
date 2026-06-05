import streamlit as st
import asyncio
import os
import time

from configs.config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION, RERANKER_MODEL
from arxiv_scholar.retrieval.orchestrator import Orchestrator
from arxiv_scholar.llm.service import LLMService

st.set_page_config(page_title="ArXiv Scholar", page_icon="📚", layout="wide")

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    /* Dark mode premium styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .source-box {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .source-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .source-title {
        font-weight: bold;
        color: #4da6ff;
        margin-bottom: 8px;
        font-size: 1.1em;
    }
    .source-title a {
        color: #4da6ff;
        text-decoration: none;
    }
    .source-title a:hover {
        text-decoration: underline;
    }
    .source-score {
        font-size: 0.85em;
        color: #aaa;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_backend():
    orchestrator = Orchestrator(
        collection_name=QDRANT_COLLECTION,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        reranker_model_name=RERANKER_MODEL
    )
    llm_service = LLMService()
    return orchestrator, llm_service

try:
    orchestrator, llm_service = load_backend()
except Exception as e:
    st.error(f"Failed to load backend Orchestrator/LLM: {e}")
    st.stop()

# Initialize session state for persistent chat & sources
if "messages" not in st.session_state:
    st.session_state.messages = []
if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

# Sidebar for Sources Navigation
with st.sidebar:
    st.title("📄 Active Sources")
    st.write("Retrieved via ML Router & BGE Re-ranker")
    st.markdown("---")
    
    if not st.session_state.latest_sources:
        st.info("Ask a research question to retrieve relevant papers.")
    else:
        for i, chunk in enumerate(st.session_state.latest_sources):
            score = chunk.get("score", 0.0)
            arxiv_id = chunk["metadata"].get("arxiv_id", "Unknown ID")
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id != "Unknown ID" else "#"
            
            st.markdown(f"""
            <div class="source-box">
                <div class="source-title"><a href="{url}" target="_blank">Source {i+1}: arXiv:{arxiv_id}</a></div>
                <div class="source-score">Relevance Match: {score:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander("View Extracted Chunk Text"):
                st.markdown(chunk["text"])

# Main Chat Interface
st.title("📚 ArXiv Scholar Assistant")
st.write("Search and chat with millions of academic physics and ML papers. Equations and LaTeX render natively!")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("What is the impact of dropout on transformers?"):
    start_time = time.time()
    
    # Append user prompt and render
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process assistant response
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("🔍 Routing query and retrieving/re-ranking academic papers...")
        
        # 1. Retrieval & Re-ranking
        try:
            chunks = asyncio.run(orchestrator.retrieve(prompt, limit=5, use_reranker=True))
            st.session_state.latest_sources = chunks
        except Exception as e:
            status_placeholder.error(f"Retrieval failed: {e}")
            st.stop()
            
        if not chunks:
            status_placeholder.warning("No relevant papers found in the database.")
            st.stop()
            
        status_placeholder.info("🧠 Formulating contextualized academic answer...")
        
        # 2. Context Building
        context_blocks = []
        for i, chunk in enumerate(chunks):
            arxiv_id = chunk["metadata"].get("arxiv_id")
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"Unknown Source {i+1}"
            context_blocks.append(f"Context {i+1} (Source: {url}):\n{chunk['text']}")
        context_str = "\n\n".join(context_blocks)
        
        # 3. Stream LLM Response
        response_placeholder = st.empty()
        
        async def generate_response():
            stream = llm_service.stream_synthesis(prompt, context_str)
            full_response = ""
            try:
                async for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        # Typing effect with cursor
                        response_placeholder.markdown(full_response + "▌")
            except Exception as e:
                response_placeholder.error(f"LLM Generation failed: {e}")
            
            # Final output without cursor
            response_placeholder.markdown(full_response)
            return full_response
            
        full_response = asyncio.run(generate_response())
        
        elapsed_time = time.time() - start_time
        time_msg = f"\n\n*⏱️ Request completed in {elapsed_time:.2f} seconds*"
        full_response += time_msg
        response_placeholder.markdown(full_response)
        
        # Clear status indicator
        status_placeholder.empty()
        
        # Save assistant message to state
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Rerun to update sidebar with the new sources
        st.rerun()
