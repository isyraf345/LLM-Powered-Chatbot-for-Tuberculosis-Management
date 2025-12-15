import streamlit as st
from dpr_retriever import DenseRetriever
from nli_verifier import NLIVerifier
from generator import AnswerGenerator
from config import DPR_MODELS


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

    # -----------------------------
    # DPR MODEL DROPDOWN (UPDATED)
    # -----------------------------
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

    question = st.text_input("Ask a TB-related question:")

    if st.button("Get Answer") and question.strip():

        with st.spinner("Retrieving relevant TB information..."):
            passages = retriever.search(question, k=5)

        st.subheader("🔍 Top Retrieved Passages")
        for idx, p in enumerate(passages):
            with st.expander(f"Passage {idx+1} (score={p['score']:.3f})"):
                st.write(f"Source: {p['source']} (page {p['page']})")
                st.write(p["text"])

        # Build GPT context
        context = build_context(passages)

        with st.spinner("Generating answer..."):
            answer = generator.generate(question, context)

        st.subheader("🤖 Chatbot Answer")
        st.write(answer)

        # NLI Verification
        with st.spinner("Running NLI Verification..."):
            texts = [p["text"] for p in passages]
            final_label, entail_score = verifier.verify_against_passages(answer, texts)

        st.subheader("🛡 NLI Verification")
        st.metric("Prediction", final_label.capitalize())
        st.metric("Entailment Score", f"{entail_score:.3f}")

        st.divider()
        st.caption("This is an educational tool, not a medical diagnosis system.")

if __name__ == "__main__":
    main()