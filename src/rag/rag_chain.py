from __future__ import annotations
import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def _format_docs(docs) -> str:
    def _src(md: dict) -> str:
        return str(md.get("source", ""))
    return "\n\n".join(f"[{_src(d.metadata)}]\n{d.page_content}" for d in docs)

def _backend() -> str:
    # "auto" usa OpenAI si hay OPENAI_API_KEY; si no, usa Ollama
    return os.getenv("RAG_BACKEND", "auto").lower()

def make_embeddings():
    be = _backend()
    if be == "openai" or (be == "auto" and os.getenv("OPENAI_API_KEY")):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model)

def make_llm():
    be = _backend()
    temperature = float(os.getenv("TEMPERATURE", "0"))
    if be == "openai" or (be == "auto" and os.getenv("OPENAI_API_KEY")):
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=temperature)
    else:
        from langchain_community.chat_models import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        return ChatOllama(model=model, temperature=temperature)

def build_vectorstore(
    data_dir: str = "./data",
    persist_dir: str = "./.vectorstore",
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> FAISS:
    docs = []
    if os.path.isdir(data_dir):
        pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True)
        docs.extend(pdf_loader.load())
        txt_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, recursive=True,
                                     loader_kwargs={"encoding": "utf-8"})
        docs.extend(txt_loader.load())
        md_loader = DirectoryLoader(data_dir, glob="**/*.md", loader_cls=TextLoader, recursive=True,
                                    loader_kwargs={"encoding": "utf-8"})
        docs.extend(md_loader.load())

    if not docs:
        raise RuntimeError(f"No se encontraron documentos en {data_dir}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    embeddings = make_embeddings()
    vs = FAISS.from_documents(documents=splits, embedding=embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    vs.save_local(persist_dir)
    return vs

def load_vectorstore(persist_dir: str = "./.vectorstore") -> FAISS:
    embeddings = make_embeddings()
    vs = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return vs

def make_retriever(vs: FAISS, k: int = 4, fetch_k: int = 20):
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})

def make_rag_chain(vs: FAISS):
    retriever = make_retriever(vs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Eres un asistente que responde de forma concisa y cita fragmentos relevantes del contexto."),
            ("human", "Pregunta: {question}\n\nContexto:\n{context}"),
        ]
    )
    chain = {
        "context": retriever | RunnableLambda(_format_docs),
        "question": RunnablePassthrough(),
    } | prompt | make_llm() | StrOutputParser()
    return chain
