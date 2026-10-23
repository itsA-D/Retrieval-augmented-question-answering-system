"""Query-time retrieval: load the persisted Chroma index and fetch top-K
nodes for a question, with a sufficiency gate to detect unsupported queries.

A vector store always returns *something* for any query, even when nothing
relevant exists. The sufficiency gate compares the best similarity score
against RETRIEVAL_THRESHOLD so downstream code can distinguish "found real
evidence" from "found the least-bad match, which is still irrelevant".
"""

import logging

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src import config

logger = logging.getLogger(__name__)

_retriever: VectorIndexRetriever | None = None


def get_retriever() -> VectorIndexRetriever:
    """Lazily load the persisted index and build a retriever (singleton)."""
    global _retriever
    if _retriever is not None:
        return _retriever

    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    chroma_collection = chroma_client.get_or_create_collection(config.CHROMA_COLLECTION)
    if chroma_collection.count() == 0:
        raise RuntimeError(
            "Chroma collection is empty. Run `python -m src.ingest` before querying."
        )

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    _retriever = VectorIndexRetriever(index=index, similarity_top_k=config.TOP_K)
    return _retriever


def retrieve(question: str) -> list[NodeWithScore]:
    retriever = get_retriever()
    results = retriever.retrieve(question)
    for r in results:
        logger.info(
            "Retrieved %-28s score=%.3f", r.node.metadata.get("source_id"), r.score or 0.0
        )
    return results


def is_sufficient(results: list[NodeWithScore]) -> bool:
    if not results:
        return False
    top_score = results[0].score or 0.0
    return top_score >= config.RETRIEVAL_THRESHOLD
