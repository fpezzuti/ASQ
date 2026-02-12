from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class OutputPaths:
    base_dir: Path
    answers: Path
    iter_queries: Path
    retrieved_docs: Path
    thoughts: Path


def build_output_paths(
    data_dir: str,
    dataset: str,
    retriever: str,
    retriever_k: int,
    agent_family: str,
    agent_generator: str,
    reranker: Optional[str] = None,
    reranker_k: Optional[int] = None,
) -> OutputPaths:

    retriever_str = f"{retriever}_k{retriever_k}"
    reranker_str = f"{reranker}_k{reranker_k}" if reranker and reranker_k else ""
    base_dir = (
        Path(data_dir) /
        "traces" /
        dataset /
        f"{retriever_str}_{reranker_str}".strip("_") /
        agent_family /
        agent_generator
    )

    answers = base_dir / "answers.tsv"
    iter_queries = base_dir / "iter_queries.tsv"
    retrieved_docs = base_dir / "retrieved_docs.tsv"
    thoughts = base_dir / "thoughts.tsv"

    return OutputPaths(base_dir=base_dir, answers=answers, iter_queries=iter_queries,
                       retrieved_docs=retrieved_docs, thoughts=thoughts)
