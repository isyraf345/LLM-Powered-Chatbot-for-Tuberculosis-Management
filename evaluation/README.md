# Evaluation preparation

`smoke_questions.json` contains nine representative prompts for regression capture,
including greetings, unsupported questions, an unrelated question, and an ambiguous
follow-up. Their retrieval outputs are observations, not correctness labels.

`relevance_labels.json` preserves all 99 original CSV examples and adds explicit
review fields. They are intentionally unreviewed. The original expected text may
be paraphrased and is not a reliable passage identifier.

To produce a defensible retrieval benchmark:

1. Inspect the saved passage corpus in `models/tb_index_single_nq/tb_passages.json`.
2. Review each question and assign relevant `id` values to `relevant_passage_ids`.
3. Set `reviewed` to true only after review. A smaller fully reviewed subset can be
   saved as a separate labels file. Keep development and final evaluation sets separate.
4. Exclude unanswerable cases from this relevance-only benchmark and assess their
   abstention behavior separately. Do not invent relevant IDs to make them pass.
5. Run from the project root:

```powershell
.\venv\Scripts\python.exe -B app/evaluate_dpr.py --labels evaluation/reviewed_subset.json --model single_nq
```

The evaluator rejects unreviewed/empty labels and unknown IDs before loading models.
It reports Hit@1/3/5 and MRR@5 using original retrieval order, with zero for misses.
These metrics assess retrieval, not medical accuracy or generated-answer quality.
Identical source/page/text passages are treated as equivalent IDs because the
current serving retriever does not return IDs. All current corpora are identical.

The previous evaluator ranked the first retrieved result by its similarity to
expected text, which made Top-5 always pass. Its results must not be reused.
