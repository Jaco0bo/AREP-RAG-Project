from __future__ import annotations

import os
import pytest

from rag.rag_chain import build_vectorstore, load_vectorstore, make_rag_chain

def test_end_to_end(tmp_path, monkeypatch):
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY no configurada")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    persist_dir = tmp_path / ".vectorstore"
    vs = build_vectorstore(str(data_dir), str(persist_dir))
    assert persist_dir.exists()

    vs2 = load_vectorstore(str(persist_dir))
    chain = make_rag_chain(vs2)
    out = chain.invoke("¿De qué trata el archivo de ejemplo?")
    assert isinstance(out, str) and len(out) > 0
