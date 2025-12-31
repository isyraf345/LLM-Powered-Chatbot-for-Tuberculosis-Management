import streamlit as st
from dpr_retriever import DenseRetriever
from generator import AnswerGenerator
from config import DPR_MODELS

# =============================
# Session state
# =============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================
# Cached loaders
# =============================
@st.cache_resource
def load_retriever(model_name: str):
    return DenseRetriever(model_name)

@st.cache_resource
def load_generator():
    return AnswerGenerator()

# =============================
# Build context for LLM
# =============================
def build_context(passages, max_chars=3500):
    blocks = []
    total = 0
    for p in passages:
        block = f"[{p['source']} - page {p['page']}]\n{p['text']}\n\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "".join(blocks)

# =============================
# Main app
# =============================
def main():
    st.set_page_config(
        page_title="TB Management Chatbot",
        page_icon="🫁",
        layout="wide"
    )

    # =============================
    # Custom UI Theme (CSS) - Updated for a more vibrant and interesting look
    # =============================
    st.markdown("""
    <style>
    /* Main background - Soft teal for a fresh, health-inspired feel */
    .stApp {
        background-color: #e0f7fa;
    }

    /* Title - Vibrant teal */
    h1 {
        color: #00695c;
        font-weight: 700;
    }

    /* Headers - Deep teal */
    h2, h3 {
        color: #00796b;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #004d40;
    }

    .stTabs [aria-selected="true"] {
        background-color: #b2dfdb;
        border-radius: 8px;
        color: #004d40;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: white;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* User message - Vibrant blue accent */
    .stChatMessage[data-testid="chat-message-user"] {
        border-left: 6px solid #0288d1;
    }

    /* Assistant message - Fresh green accent */
    .stChatMessage[data-testid="chat-message-assistant"] {
        border-left: 6px solid #009688;
    }

    /* Sidebar - Light cyan */
    section[data-testid="stSidebar"] {
        background-color: #b2ebf2;
    }

    /* Buttons - Vibrant teal */
    .stButton > button {
        background-color: #00acc1;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s;
    }

    .stButton > button:hover {
        background-color: #00838f;
    }

    /* Expanders */
    details summary {
        font-weight: 600;
        color: #00796b;
    }

    /* Info box - Soft teal */
    div[data-testid="stAlert"] {
        background-color: #e0f2f1;
        border-radius: 10px;
        border-left: 5px solid #009688;
    }
    </style>
    """, unsafe_allow_html=True)


    st.title("🫁 LLM-Powered Tuberculosis Management System")

    # =============================
    # Tabs
    # =============================
    tab1, tab2, tab3 = st.tabs([
        "📘 Basic Tuberculosis Information",
        "💬 Tuberculosis Chatbot",
        "🧠 System Overview"
    ])

    # =========================================================
    # TAB 1 — BASIC TB INFORMATION
    # =========================================================
    with tab1:
        st.header("📘 Basic Information About Tuberculosis")

        st.markdown("""
        Tuberculosis (TB) is a contagious infectious disease caused by  
        *Mycobacterium tuberculosis*. It mainly affects the lungs (pulmonary TB)  
        but can also affect other organs (extrapulmonary TB).

        ### 🔹 Key Facts
        - TB spreads through airborne droplets
        - Common symptoms:
          - Persistent cough (≥2 weeks)
          - Fever
          - Night sweats
          - Weight loss
        - TB is curable and preventable

        ### 🔹 Types of TB
        - Latent TB Infection (LTBI) – infected but not sick or contagious
        - Active TB Disease – symptomatic and infectious
        - Drug-Resistant TB (DR-TB) – resistant to standard treatment

        ### 🔹 Treatment
        - Standard TB treatment lasts 6 months
        - Drug-resistant TB requires longer, specialized regimens

        ### 🔹 Sources
        - World Health Organization (WHO)
        - Ministry of Health Malaysia (MOH)
        """)

        st.info(
            "This section provides general educational information and does not replace professional medical advice."
        )

    # =========================================================
    # TAB 2 — CHATBOT
    # =========================================================
    with tab2:
        st.header("💬 Ask the Tuberculosis Chatbot")
        st.caption("DPR + FAISS + GPT (RAG-based QA)")

        # Sidebar
        st.sidebar.header("⚙️ Model Settings")
        model_choice = st.sidebar.selectbox(
            "Choose DPR Retriever Model:",
            list(DPR_MODELS.keys()),
            format_func=lambda x: {
                "single_nq": "Single NQ (Facebook DPR)",
                "multiset": "Multiset (Facebook DPR)",
                "lfqa": "LFQA DPR (Long-Form QA)"
            }.get(x, x)
        )

        retriever = load_retriever(model_choice)
        generator = load_generator()

        # =============================
        # Render chat history
        # =============================
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])

            with st.chat_message("assistant"):
                st.write(turn["answer"])

                with st.expander("🔍 Retrieved Passages"):
                    for idx, p in enumerate(turn["passages"]):
                        st.markdown(f"**Passage {idx + 1}** (score={p['score']:.3f})")
                        st.write(f"Source: {p['source']} (page {p['page']})")
                        st.write(p["text"])
                        st.divider()

        # =============================
        # Chat input (ALWAYS LAST)
        # =============================
        question = st.chat_input("Ask a TB-related question...")

        if question:
            with st.spinner("Retrieving relevant TB information..."):
                passages = retriever.search(question, k=5)

            context = build_context(passages)

            with st.spinner("Generating answer..."):
                answer = generator.generate(question, context)

            # Save only (no rendering here)
            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "passages": passages
            })

            # Force clean rerun so input stays at bottom
            st.rerun()

        st.caption("⚠️ Educational use only. Not a medical diagnosis system.")

    # =========================================================
    # TAB 3 — SYSTEM OVERVIEW
    # =========================================================
    with tab3:
        st.header("🧠 System Overview & Model Explanation")

        st.markdown("""
        ### 🔹 Overall System Architecture
        This project is built using a Retrieval-Augmented Generation (RAG) framework.
        Instead of relying purely on a language model’s internal knowledge, the system
        retrieves relevant medical documents first, then generates answers grounded
        in authoritative sources.

        ---
        ### 🔹 Dense Passage Retrieval (DPR)
        Purpose: Retrieve the most relevant passages from TB-related documents.

        Why DPR?
        - Traditional keyword search (e.g., TF-IDF, BM25) struggles with medical phrasing.
        - DPR encodes both questions and passages into dense vectors, allowing
        semantic matching.
                    
        ### Dense Passage Retrieval (DPR) Variants Used

        Facebook DPR model trained on Natural Questions Dataset (single_nq)
        Trained solely on the Natural Questions dataset, this model excels at retrieving highly relevant passages for concise, fact-based questions. It serves as a strong baseline for precise information retrieval.

        Facebook DPR model trained on Multiple datasets (Multiset)
        Trained on multiple question-answer datasets, Multiset DPR improves robustness and generalization across diverse question styles, making it suitable for less structured or user-generated queries.

        Long-Form Question Answering DPR (LFQA)
        Optimized for long-form question answering, this model retrieves passages that collectively support detailed explanations. It is particularly effective for medical questions that require multi-sentence responses.

        How it works:
        1. User question is converted into a dense vector
        2. All document passages are pre-encoded into vectors
        3. Similarity is computed using inner product search

        Benefit:  
        Retrieves medically relevant passages even when wording differs
        (e.g., synonyms, paraphrasing).

        ---
        ### 🔹 FAISS
        Purpose: Efficient similarity search over large numbers of embeddings.
        Why FAISS?
        - Optimized for fast nearest-neighbor search
        - Scales well with thousands of document chunks
        - Industry-standard for vector search systems

        Role in system:
        - Stores DPR passage embeddings
        - Retrieves Top-K most relevant passages for each query

        ---
        ### 🔹 Large Language Model (gpt-4o-mini)
        Purpose: Generate human-readable answers.

        ---
        ### 🔹 Intended Use
        This system is designed for:
        - Educational purposes
        - Medical information support
        - Demonstration of modern RAG-based AI systems

        ⚠️ This system is not a medical diagnosis tool.
        """)

# =============================
# Run app
# =============================
if __name__ == "__main__":
    main()

