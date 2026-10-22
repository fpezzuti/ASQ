import pyterrier as pt

from .gpt_oss import GptOssAgent


def build_gpt_oss(
    retrieval_pipeline: pt.Transformer,
    model_id: str,
    agent_k: int,
    retriever_batch_size: int = 1,
    retriever_batch_wait_ms: int = 25,
):
    return GptOssAgent(
        retriever=retrieval_pipeline,
        model_id=model_id,
        agent_k=agent_k,
        retriever_batch_size=retriever_batch_size,
        retriever_batch_wait_ms=retriever_batch_wait_ms,
    )
