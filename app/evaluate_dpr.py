import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List

from dpr_retriever import DenseRetriever
from config import DPR_MODELS

from sentence_transformers import SentenceTransformer, util


# -----------------------------------------------------------
# Metrics
# -----------------------------------------------------------

def reciprocal_rank(ranks: List[int]):
    rr = 0
    for r in ranks:
        if r != -1:
            rr += 1.0 / (r + 1)
    return rr / len(ranks)


# -----------------------------------------------------------
# Evaluation Function
# -----------------------------------------------------------

def evaluate_model(model_name: str, df: pd.DataFrame, embedder):
    retriever = DenseRetriever(model_name)

    print(f"\n===============================")
    print(f"Evaluating DPR model: {model_name}")
    print(f"===============================")

    top1, top3, top5 = 0, 0, 0
    ranks = []
    cos_sims = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        question = row["question"]
        expected_passage = row["expected_passage"]

        # Retrieve passages
        retrieved = retriever.search(question, k=5)
        retrieved_texts = [p["text"] for p in retrieved]

        # --- Embed using SentenceTransformer ---
        embeddings = embedder.encode([expected_passage] + retrieved_texts, convert_to_tensor=True)

        gt_emb = embeddings[0]
        retrieved_embs = embeddings[1:]

        # Compute cosine similarities
        similarities = util.cos_sim(gt_emb, retrieved_embs)[0].cpu().numpy()

        # Save average cosine similarity for reporting
        cos_sims.append(float(np.max(similarities)))

        # Determine rank by semantic similarity
        sorted_idx = np.argsort(similarities)[::-1]  # highest → lowest
        rank = int(np.where(sorted_idx == 0)[0][0]) if 0 < len(embeddings) else -1
        ranks.append(rank)

        # Accuracy metrics
        if rank == 0:
            top1 += 1
        if rank < 3:
            top3 += 1
        if rank < 5:
            top5 += 1

    # Final metrics
    n = len(df)
    print(f"\nModel: {model_name}")
    print(f"Top-1 Accuracy: {top1 / n:.3f}")
    print(f"Top-3 Accuracy: {top3 / n:.3f}")
    print(f"Top-5 Accuracy: {top5 / n:.3f}")
    print(f"MRR: {reciprocal_rank(ranks):.3f}")
    print(f"Average Semantic Cosine Similarity: {np.mean(cos_sims):.3f}")
    print("Done.\n")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    print("Loading QA dataset...")
    df = pd.read_csv("qa-pairs.csv")

    print(f"Loaded {len(df)} QA pairs.")

    print("Loading SentenceTransformer model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Evaluate all DPR models
    for model_name in DPR_MODELS.keys():
        evaluate_model(model_name, df, embedder)


if __name__ == "__main__":
    main()
