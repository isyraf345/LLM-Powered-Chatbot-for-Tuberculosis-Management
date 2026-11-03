"""Exercise actual API/retrieval with cached CPU models and a recording fake generator.

No provider calls; never overwrites the Phase 1 baseline.
"""
import json
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-local', action='store_true', help='Exercise enabled tracing with an in-memory exporter; never contacts LangSmith.')
    args = parser.parse_args()
    import numpy as np
    from fastapi.testclient import TestClient
    from app.api import create_app
    from app.dpr_retriever import DenseRetriever
    from app.evaluate_dpr import resolve_ids
    from app.service import ChatService
    from app.observability import Tracer

    class Recorder:
        def generate(self, question, context):
            self.context = context
            return 'Offline verification only; no generated medical answer.'

        def close(self):
            pass

    class LocalExporter:
        def __init__(self):
            self.records = []

        def create_run(self, **record):
            self.records.append(record)

        def close(self):
            pass

    exporter = LocalExporter()
    tracer = Tracer(enabled=args.trace_local, client_factory=lambda: exporter)
    recorder = Recorder()
    service = ChatService(lambda name: DenseRetriever(name, device='cpu'), lambda: recorder, tracer=tracer)
    corpus = json.loads((ROOT / 'models/tb_index_single_nq/tb_passages.json').read_text(encoding='utf-8'))
    total = 0
    with TestClient(create_app(lambda: service)) as client:
        for model in ('single_nq', 'multiset', 'lfqa'):
            baseline = json.loads((ROOT / f'docs/baseline/retrieval-{model}.json').read_text(encoding='utf-8'))
            for example in baseline['examples']:
                response = client.post('/chat', json={'question': example['question'], 'retriever': model})
                assert response.status_code == 200, response.text
                passages = response.json()['passages']
                assert [sorted(ids) for ids in resolve_ids(passages, corpus)] == example['retrieved_ids'], example['id']
                np.testing.assert_allclose([p['score'] for p in passages], example['scores'], atol=1e-6, rtol=1e-5)
                assert recorder.context == example['context'], example['id']
                total += 1
            print(f'{model}: all {len(baseline["examples"])} API requests match baseline', flush=True)
        if args.trace_local:
            from time import monotonic, sleep
            deadline = monotonic() + 3
            while tracer.exported < total and monotonic() < deadline:
                sleep(.01)
            assert tracer.exported == total, tracer.status()
            assert len(exporter.records) == total * 5
            print(f'Local tracing: {total} parent traces and {total * 4} child runs exported.')
    print(f'PASS: {total} API requests; real retrieval and exact context; generation mocked.')


if __name__ == '__main__':
    main()
