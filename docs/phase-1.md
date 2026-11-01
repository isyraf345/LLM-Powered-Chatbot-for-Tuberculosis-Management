# Phase 1 baseline and preparation

## Preserved

- Live Streamlit request flow and context builder.
- All three DPR configurations, indexes, and passage files.
- Generation prompt, model, temperature, top-p, and output limit.
- Original 99-row CSV and project reports.
- Existing staged Git changes; nothing was committed, staged, reset, or unstaged.

`baseline/manifest.json` records the original hashes and secret-free configuration.
`baseline/installed-environment.txt` is an observed installed-package inventory,
not a cross-platform lockfile. Direct requirements are pinned to installed versions.

## Changes

- API key now comes from the environment; a missing key produces an explicit error.
- Restored Git ignore rules and original LFS attribute rules. Added secret/output
  exclusions. Index files remain eligible for version control.
- Archived the original notebook unchanged as `experiments/tb_legacy.ipynb`.
- Removed the disconnected application NLI module in the preceding cleanup.
- Replaced flawed evaluation ranks with reviewed-ID relevance metrics and tests.
- Prepared nine regression prompts and a 99-example annotation worksheet.

## Validation scope and remaining work

- All three cached DPR variants completed CPU retrieval for nine smoke questions
  each (27 captured requests), without downloads or generation API calls.
- Five retrieval-metric tests passed; `pip check` reported no broken requirements.
- Parsed all application/script/test Python files and all baseline JSON artifacts.
- Verified unchanged hashes for the live UI, retriever, six model artifacts, original
  CSV, and archived notebook. The generator changed only to reject a missing key.
- Confirmed missing-key startup and rejection of unreviewed evaluation labels.
- Model loading emitted tokenizer-class and unused-pooler-weight warnings. Capture
  succeeded; these existing warnings should be investigated during model-loading
  cleanup rather than silently changing tokenization in this baseline phase.
- Retrieval snapshots capture actual local search/context behavior, not answer accuracy.
- No paid generation calls were made. Generated-answer baseline capture remains
  pending a newly rotated key. Do not claim end-to-end parity from retrieval alone.
- Human relevance annotation is still required; no evaluation accuracy is fabricated.
- Key rotation/revocation requires the account owner. Removing a key from a file
  does not revoke it or erase it from earlier copies/history.
- Original source PDFs/final indexing workflow still need recovery for index rebuilds.
- Fresh-environment installation and full Streamlit UI smoke testing remain unverified.
- GitHub integration preserved its corrected GPT-4o-mini UI label and portable
  model paths. The baseline manifest describes the earlier local snapshot.
- The empty `app/qa-pairs.txt` is retained to avoid unrelated cleanup.

The next phase is API extraction. It should preserve the captured retrieval outputs
before introducing conversation memory or image-aware behavior.
