# dpr_retriever.py

import json
from typing import List, Dict

import torch
import faiss
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast
)

from config import DPR_MODELS


class DenseRetriever:
    def __init__(self, model_name: str, device: str = None):
        """
        model_name must be: "single_nq", "multiset", or "lfqa"
        """
        if model_name not in DPR_MODELS:
            raise ValueError(f"❌ Unknown DPR model '{model_name}'. "
                             f"Available: {list(DPR_MODELS.keys())}")

        self.model_name = model_name
        cfg = DPR_MODELS[model_name]

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n==============================")
        print(f"🔍 Loading DPR Model: {model_name}")
        print("==============================\n")

        # -----------------------------
        # Load tokenizers
        # -----------------------------
        print("📥 Loading tokenizers...")
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(cfg["tokenizer"])
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(cfg["tokenizer"])

        # -----------------------------
        # Load DPR Encoders
        # -----------------------------
        print("📥 Loading context encoder:", cfg["ctx"])
        self.ctx_model = DPRContextEncoder.from_pretrained(cfg["ctx"]).to(self.device)
        self.ctx_model.eval()

        print("📥 Loading question encoder:", cfg["query"])
        self.q_model = DPRQuestionEncoder.from_pretrained(cfg["query"]).to(self.device)
        self.q_model.eval()

        # -----------------------------
        # Load FAISS index
        # -----------------------------
        print("📁 Loading FAISS index:", cfg["index"])
        self.index = faiss.read_index(cfg["index"])

        # -----------------------------
        # Load passages
        # -----------------------------
        print("📄 Loading passages:", cfg["passages"])
        with open(cfg["passages"], "r", encoding="utf-8") as f:
            self.passages = json.load(f)

        print(f"✅ Loaded {len(self.passages)} passages for model '{model_name}'\n")

    # ------------------------------------------------------------------
    # Encode user query
    # ------------------------------------------------------------------
    def _encode_query(self, query: str):
        enc = self.q_tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.q_model(**enc)
            pooled = outputs.pooler_output
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        return pooled.cpu().numpy().astype("float32")

    # ------------------------------------------------------------------
    # Retrieve top-k passages
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> List[Dict]:
        q_emb = self._encode_query(query)
        scores, idxs = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            p = self.passages[int(idx)]
            results.append({
                "score": float(score),
                "text": p["text"],
                "source": p["source"],
                "page": p.get("page", None),
            })

        return results
