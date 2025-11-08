from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from rag.rag_chain import load_vectorstore, make_rag_chain

load_dotenv()

app = FastAPI(title="RAG LangChain API")

# carga perezosa en el primer request
_chain = None

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query):
    global _chain
    if _chain is None:
        vs = load_vectorstore("./.vectorstore")
        _chain = make_rag_chain(vs)
    result = _chain.invoke(q.question)
    return {"answer": result}
