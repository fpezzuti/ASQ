import pyterrier as pt

from trace_generation.rag_agent.search_r1.search_r1 import SearchR1


def build_autorefine(retrieval_pipeline: pt.Transformer, model_id: str, agent_k: int):
    return SearchR1(retriever=retrieval_pipeline, agent_k=agent_k, model_id=model_id)
