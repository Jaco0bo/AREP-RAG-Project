from __future__ import annotations

import argparse
from .rag_chain import build_vectorstore

def main():
    parser = argparse.ArgumentParser(description="Ingesta/actualización del índice FAISS.")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--persist_dir", type=str, default="./.vectorstore")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    args = parser.parse_args()

    vs = build_vectorstore(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"✅ Índice FAISS guardado en {args.persist_dir}")

if __name__ == "__main__":
    main()
