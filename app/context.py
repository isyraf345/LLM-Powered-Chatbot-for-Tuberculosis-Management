"""Context construction shared by the API and the reference Streamlit interface."""


def build_context(passages, max_chars=3500):
    blocks = []
    total = 0
    for p in passages:
        block = f"[{p['source']} - page {p['page']}]\n{p['text']}\n\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "".join(blocks)
