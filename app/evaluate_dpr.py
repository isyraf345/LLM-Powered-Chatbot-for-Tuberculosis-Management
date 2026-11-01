"""Evaluate retrieval against reviewed passage IDs; never infer relevance from similarity."""
import argparse
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def score_retrieval(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Reviewed relevant passage IDs are required.")
    rank = next((i for i, ids in enumerate(retrieved_ids, 1) if relevant.intersection(ids)), None)
    return {"hit_at_1": float(rank is not None and rank <= 1),
            "hit_at_3": float(rank is not None and rank <= 3),
            "hit_at_5": float(rank is not None and rank <= 5),
            "mrr_at_5": 1.0 / rank if rank is not None and rank <= 5 else 0.0}


def passage_key(p):
    return p['source'], p.get('page'), p['text']


def resolve_ids(passages, corpus):
    lookup = {}
    for p in corpus:
        lookup.setdefault(passage_key(p), set()).add(p['id'])
    return [lookup[passage_key(p)] for p in passages]


def validate_labels(rows, valid_ids):
    if not rows:
        raise ValueError("Evaluation set is empty.")
    seen = set()
    for row in rows:
        if row['id'] in seen:
            raise ValueError("Duplicate example ID.")
        seen.add(row['id'])
        ids = row.get('relevant_passage_ids')
        if row.get('reviewed') is not True or not ids:
            raise ValueError(f"{row['id']}: label relevant_passage_ids and set reviewed=true first.")
        if not set(ids).issubset(valid_ids):
            raise ValueError(f"{row['id']}: unknown passage ID.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'evaluation/relevance_labels.json')
    parser.add_argument('--model', choices=['single_nq', 'multiset', 'lfqa'], default='single_nq')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/retrieval-evaluation.json')
    args = parser.parse_args()
    rows = json.loads(args.labels.read_text(encoding='utf-8'))
    corpus = json.loads((ROOT / 'models/tb_index_single_nq/tb_passages.json').read_text(encoding='utf-8'))
    try:
        validate_labels(rows, {p['id'] for p in corpus})
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output.resolve()
    # Retain current retriever paths until the API refactor.
    os.chdir(ROOT / 'app')
    from dpr_retriever import DenseRetriever
    retriever = DenseRetriever(args.model)
    results = []
    for row in rows:
        passages = retriever.search(row['question'], k=5)
        ids = resolve_ids(passages, retriever.passages)
        results.append({'id': row['id'], 'retrieved_ids': [sorted(x) for x in ids],
                        **score_retrieval(ids, row['relevant_passage_ids'])})
    metrics = {k: sum(r[k] for r in results) / len(results)
               for k in ('hit_at_1', 'hit_at_3', 'hit_at_5', 'mrr_at_5')}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'model': args.model, 'count': len(rows), 'metrics': metrics,
                                  'examples': results}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
