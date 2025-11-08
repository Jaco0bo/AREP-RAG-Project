from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv

from .rag_chain import load_vectorstore, make_rag_chain

def main():
    parser = argparse.ArgumentParser(description="Chat CLI sobre tu índice RAG.")
    parser.add_argument("--persist_dir", type=str, default="./.vectorstore")
    args = parser.parse_args()

    load_dotenv()
    vs = load_vectorstore(args.persist_dir)
    chain = make_rag_chain(vs)

    print("💬 RAG listo. Escribe 'exit' para salir.")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            answer = chain.invoke(q)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
