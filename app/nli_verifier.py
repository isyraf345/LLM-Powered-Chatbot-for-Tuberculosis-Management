# nli_verifier.py

from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import NLI_MODEL_NAME


class NLIVerifier:
    """
    Biomedical NLI verifier using the correct label mapping for:
    Bam3752/PubMedBERT-BioNLI-LoRA

    Label order:
      0 = neutral
      1 = contradiction
      2 = entailment
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading NLI model: {NLI_MODEL_NAME}")

        self.tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(self.device)
        self.model.eval()

        # CORRECT label mapping
        self.index_to_label = {
            0: "neutral",
            1: "contradiction",
            2: "entailment",
        }

    def classify(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Return predicted NLI label + entailment score"""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**inputs)
            probs = F.softmax(out.logits, dim=-1)[0].cpu()

        pred_idx = int(torch.argmax(probs).item())
        pred_label = self.index_to_label[pred_idx]

        # CORRECT: entailment = index 2
        entail_score = float(probs[2])

        return pred_label, entail_score

    def verify_against_passages(self, answer: str, passages: List[str]) -> Tuple[str, float]:
        """
        Best-match strategy:
        - use the passage with highest entailment probability
        - ensures more stable verification
        """
        if not passages:
            return "neutral", 0.0

        best_entail_score = 0.0
        best_label = "neutral"

        for passage in passages[:3]:  # limit to top-3 to reduce noise
            label, score = self.classify(passage, answer)

            if score > best_entail_score:
                best_entail_score = score
                best_label = label

        # threshold-based final decision
        if best_entail_score >= 0.6:
            final_label = "entailment"
        elif best_entail_score >= 0.3:
            final_label = "neutral"
        else:
            final_label = "contradiction"

        return final_label, round(best_entail_score, 3)
