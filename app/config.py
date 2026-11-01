import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# ============================
# DPR MODEL CONFIGURATIONS
# ============================

# config.py (inside app/)

DPR_MODELS = {
    "single_nq": {
        "ctx": "facebook/dpr-ctx_encoder-single-nq-base",
        "query": "facebook/dpr-question_encoder-single-nq-base",
        "tokenizer": "facebook/dpr-ctx_encoder-single-nq-base",
        "index": str(MODELS_DIR / "tb_index_single_nq/tb_faiss.index"),
        "passages": str(MODELS_DIR / "tb_index_single_nq/tb_passages.json"),
    },

    "multiset": {
        "ctx": "facebook/dpr-ctx_encoder-multiset-base",
        "query": "facebook/dpr-question_encoder-multiset-base",
        "tokenizer": "facebook/dpr-ctx_encoder-single-nq-base",
        "index": str(MODELS_DIR / "tb_index_multiset/faiss.index"),
        "passages": str(MODELS_DIR / "tb_index_multiset/passages.json"),
    },

    "lfqa": {
        "ctx": "vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
        "query": "vblagoje/dpr-question_encoder-single-lfqa-base",
        "tokenizer": "facebook/dpr-ctx_encoder-single-nq-base",
        "index": str(MODELS_DIR / "tb_index_lfqa/faiss.index"),
        "passages": str(MODELS_DIR / "tb_index_lfqa/passages.json"),
    },

}


# ============================
# OpenAI
# ============================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = "gpt-4o-mini"
