import streamlit as st
from dpr_retriever import DenseRetriever
from nli_verifier import NLIVerifier
from generator import AnswerGenerator
from config import DPR_MODELS

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_retriever(model_name: str):
    return DenseRetriever(model_name)


@st.cache_resource
def load_verifier():
    return NLIVerifier()


@st.cache_resource
def load_generator():
    return AnswerGenerator()


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

def main():
    st.title("🫁 LLM-Powered Chatbot for Tuberculosis Management")
    st.caption("DPR + FAISS + GPT + Biomedical NLI Verification")

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
    verifier = load_verifier()
    generator = load_generator()

    # -----------------------------
    # DISPLAY CHAT HISTORY
    # -----------------------------
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])

        with st.chat_message("assistant"):
            st.write(turn["answer"])

            st.subheader("🛡 NLI Verification")
            st.metric("Prediction", turn["nli_label"].capitalize())
            st.metric("Entailment Score", f"{turn['entail_score']:.3f}")

            with st.expander("🔍 Retrieved Passages"):
                for idx, p in enumerate(turn["passages"]):
                    st.markdown(f"**Passage {idx+1}** (score={p['score']:.3f})")
                    st.write(f"Source: {p['source']} (page {p['page']})")
                    st.write(p["text"])
                    st.divider()

    # -----------------------------
    # CHAT INPUT (BOTTOM)
    # -----------------------------
    question = st.chat_input("Ask a TB-related question...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant TB information..."):
                passages = retriever.search(question, k=5)

            context = build_context(passages)

            with st.spinner("Generating answer..."):
                answer = generator.generate(question, context)

            st.write(answer)

            with st.spinner("Running NLI Verification..."):
                texts = [p["text"] for p in passages]
                final_label, entail_score = verifier.verify_against_passages(answer, texts)

            st.subheader("🛡 NLI Verification")
            st.metric("Prediction", final_label.capitalize())
            st.metric("Entailment Score", f"{entail_score:.3f}")

            with st.expander("🔍 Retrieved Passages"):
                for idx, p in enumerate(passages):
                    st.markdown(f"**Passage {idx+1}** (score={p['score']:.3f})")
                    st.write(f"Source: {p['source']} (page {p['page']})")
                    st.write(p["text"])
                    st.divider()

        # Save turn to session state
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "passages": passages,
            "nli_label": final_label,
            "entail_score": entail_score
        })

    st.caption("This is an educational tool, not a medical diagnosis system.")

if __name__ == "__main__":
    main()
