"""Capture local retrieval/context outputs without generation API calls."""
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, str(ROOT / 'app'))


def main():
    from context import build_context
    from dpr_retriever import DenseRetriever
    from evaluate_dpr import resolve_ids
    examples = json.loads((ROOT / 'evaluation/smoke_questions.json').read_text(encoding='utf-8'))
    os.chdir(ROOT / 'app')
    for model in ('single_nq', 'multiset', 'lfqa'):
        retriever = DenseRetriever(model, device='cpu')
        results = []
        for example in examples:
            passages = retriever.search(example['question'], k=5)
            results.append({**example, 'retrieved_ids': [sorted(x) for x in resolve_ids(passages, retriever.passages)],
                            'scores': [p['score'] for p in passages],
                            'context': build_context(passages)})
        target = ROOT / f'docs/baseline/retrieval-{model}.json'
        target.write_text(json.dumps({'model': model, 'device': 'cpu', 'generated_answers': False,
                                      'examples': results}, indent=2) + '\n', encoding='utf-8')
        print(f'Captured {model}: {len(results)} questions', flush=True)
        del retriever


if __name__ == '__main__':
    main()
