# Instructions for coding agents

Read `README.md` and `docs/ROADMAP.md` before changing the project. The roadmap is
the handoff for the agreed seven-phase portfolio plan; do not assume missing chat
history means the project should be redesigned.

- Inspect Git status and relevant source before editing. Preserve unrelated user
  files and changes. Root-level reports/manuals are local supporting material;
  do not automatically stage them, environments, credentials, or caches.
- Keep current retrieval/generation behavior unless the user authorizes a change.
  Preserve all three DPR variants and the existing FAISS indexes. Do not restore
  the removed application NLI verifier from the archived notebook.
- Use `python -B -m unittest discover -s tests -v` with the project environment for
  relevant Python/API changes. `scripts/verify_api_baseline.py` verifies real cached
  retrieval without provider calls; run it when retrieval/context behavior changes.
- Do not run `scripts/capture_baseline.py` as a routine test: it overwrites baseline
  snapshots. Review intended behavior changes before accepting replacement baselines.
- Never print or commit secrets. Keys come from the environment or the Git-ignored root `.env`, loaded
  automatically without overriding existing environment values. Do not assume key rotation or live validation is complete.
- Treat historical validation results as evidence from that revision, not proof
  that untested changes pass. Do not fabricate accuracy or reviewed labels.
- Keep `docs/ROADMAP.md` current after each milestone: actual changes, checks,
  unresolved work, and next steps. Keep phase notes for implementation details.

User instructions take precedence over this handoff. The seven-phase roadmap is
not blanket authorization to deploy, publish, or implement every future phase.
