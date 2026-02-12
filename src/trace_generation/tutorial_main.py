

import argparse
import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import logging
from typing import Optional

from data_utils.retrieval_utils import build_retrieval_pipeline
from trace_generation.extraction import DataFrameExtractor, ExtractionType
from trace_generation.output_paths import build_output_paths, OutputPaths
from trace_generation.rag_agent import SUPPORTED_MODEL_TYPES
from trace_generation.utils import build_agent, filter_existing_answers, load_queries, process_queries_batch, setup_logging


DATA_DIR = "./../data/" # path to the data directory where the traces are stored

BATCH_SIZE = 1


def build_extractor(extract_thoughts: bool) -> DataFrameExtractor:
    if extract_thoughts:
        extraction_order = [
            ExtractionType.ANSWERS,
            ExtractionType.THOUGHTS,
            ExtractionType.ITER_QUERIES,
            ExtractionType.RETRIEVED_DOCS,
        ]
    else:
        extraction_order = [
            ExtractionType.ANSWERS,
            ExtractionType.ITER_QUERIES,
            ExtractionType.RETRIEVED_DOCS,
        ]
    return DataFrameExtractor(extraction_order=extraction_order)


def batch_processing(
    df_queries: pd.DataFrame,
    agent,
    dataframe_extractor: DataFrameExtractor,
    output_fpaths: OutputPaths,
    max_queries: Optional[int],
    batch_size: int = BATCH_SIZE,
    logger: logging.Logger = logging.getLogger(__name__),
) -> int:
    total_queries_processed = 0
    df_queries_to_proc = filter_existing_answers(df_queries, output_fpaths.answers, logger)
    for start_idx in tqdm(range(0, len(df_queries_to_proc), batch_size), desc="Processing queries", unit="queries"):
        batch_df = df_queries_to_proc.iloc[start_idx : start_idx + batch_size]
        try:
            process_queries_batch(
                queries_df=batch_df,
                agent=agent,
                dataframe_extractor=dataframe_extractor,
                output_paths=output_fpaths,
            )
        except Exception as e:
            logger.error(f"Error: {e}")

        total_queries_processed += len(batch_df)
        if max_queries is not None and total_queries_processed >= max_queries:
            break
    return total_queries_processed

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SearchR1 answers for queries.")
    parser.add_argument("--dataset", type=str, required=True, help=f"Dataset to process. E.g., hotpot-qa, msmarco-dev, researchy-test, researchy-train.")
    parser.add_argument("--search_r1_k", type=int, default=3, help="Number of top documents to retrieve with SearchR1.")
    parser.add_argument("--model_type", type=str, default="search_r1", choices=SUPPORTED_MODEL_TYPES, help="Model backend (e.g., search_r1).")
    parser.add_argument("--model_id", type=str, default="Qwen-7B", help="Model key to use for SearchR1.")
    parser.add_argument("--wmodel", type=str, default="BM25", help="Base retrieval model (e.g., BM25, DPH, PL2).")
    parser.add_argument("--retriever_k", type=int, default=1000, help="Number of candidates retrieved by the base retriever.")
    parser.add_argument("--reranker_k", type=int, default=100, help="Number of candidates kept by the reranker.")
    parser.add_argument("--ram_limit", type=int, default=65000, help="PyTerrier Java RAM limit (MB).")
    parser.add_argument("--max_queries", type=int, default=None, help="Maximum number of queries to process (for testing).")
    parser.add_argument("--judged_queries_only", action='store_true', help="Process only queries that have judgments.")
    parser.add_argument("--extract_thoughts", action='store_true', help="Extract thoughts from the agent's output.")
    parser.add_argument("--rerank", action='store_true', help="Use reranking in the retrieval pipeline.")
    return parser



def main():
    logger = logging.getLogger(__name__)
    setup_logging(logger)
    parser = build_arg_parser()
    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    pt.java.set_memory_limit(args.ram_limit)

    try:
        df_queries = load_queries(args)
    except RuntimeError as e:
        logger.error(str(e))
        return
    
    logger.info(f"Number of unique qids: {df_queries['qid'].nunique()}")
    reranker_k = args.reranker_k if args.rerank else 0

    pipeline = build_retrieval_pipeline( # any pyterrier transformer that takes qid and query columns, and provides text, title and rank columns can be used here.
        dataset=args.dataset,
        k1=args.retriever_k,
        k2=reranker_k if args.rerank else None,
        batch_size=128,
        wmodel=args.wmodel,
    )
    

    logger.info(f"Initialising {args.model_type} (will use top {args.search_r1_k} documents).")
    agent = build_agent(args, pipeline)
    logger.info(f"Starting search for {len(df_queries)} queries.")

    output_paths = build_output_paths(
        data_dir=DATA_DIR,
        dataset=args.dataset,
        retriever=args.wmodel,
        retriever_k=args.retriever_k,
        reranker="electra" if args.rerank else "none",
        reranker_k=reranker_k,
        agent_family=args.model_type,
        agent_generator=args.model_id,
    )

    dataframe_extractor = build_extractor(args.extract_thoughts)
    batch_processing(
        df_queries=df_queries,
        agent=agent,
        dataframe_extractor=dataframe_extractor,
        max_queries=args.max_queries,
        batch_size=BATCH_SIZE,
        output_fpaths=output_paths,
        logger=logger,
    )
    logger.info("Done.")

if __name__ == "__main__":
    main()