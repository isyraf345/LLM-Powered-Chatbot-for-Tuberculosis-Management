# ============================
# DPR MODEL CONFIGURATIONS
# ============================

# config.py (inside app/)

DPR_MODELS = {
    "single_nq": {
        "ctx": "facebook/dpr-ctx_encoder-single-nq-base",
        "query": "facebook/dpr-question_encoder-single-nq-base",
        "tokenizer": "facebook/dpr-ctx_encoder-single-nq-base",
        "index": "../models/tb_index_single_nq/tb_faiss.index",
        "passages": "../models/tb_index_single_nq/tb_passages.json",
    },

    "multiset": {
        "ctx": "facebook/dpr-ctx_encoder-multiset-base",
        "query": "facebook/dpr-question_encoder-multiset-base",
        "tokenizer": "facebook/dpr-ctx_encoder-single-nq-base",
        "index": "../models/tb_index_multiset/faiss.index",
        "passages": "../models/tb_index_multiset/passages.json",
    },

    "lfqa": {
        "ctx": "vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
        "query": "vblagoje/dpr-question_encoder-single-lfqa-base",
        "tokenizer": "facebook/dpr-ctx_encoder-single-nq-base",
        "index": "../models/tb_index_lfqa/faiss.index",
        "passages": "../models/tb_index_lfqa/passages.json",
    },

}


# ============================
# NLI MODEL
# ============================

NLI_MODEL_NAME = "Bam3752/PubMedBERT-BioNLI-LoRA"

# ============================
# OpenAI
# ============================
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it in Streamlit Secrets or environment variables."
    )
OPENAI_MODEL = "gpt-4o-mini"

