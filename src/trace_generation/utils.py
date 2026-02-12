import os
import pandas as pd

from trace_generation.extraction.extractor_df import DataFrameExtractor
from data_utils import load_dataset
from artifacts.answers import load_answers


import logging
import time
import sys

def setup_logging(logger, level=logging.INFO) -> None:
    """
        Sets up the global logger with file and console handlers.
    """
    if logger.handlers: # logger already set up
        return 
        
    logger.setLevel(level) 
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f'search_r1_agent_{timestamp}.log'

    file_handler = logging.FileHandler(log_filename)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level) 
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level) 
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

BATCH_SIZE = 1

def filter_existing_answers(df_queries: pd.DataFrame, answers_path: str, logger) -> pd.DataFrame:
    """
    Filter out queries that already have answers in answer's output tsv file.
    """
    if not os.path.exists(answers_path):
        logger.info(f"Output answers file {answers_path} does not exist.")
        return df_queries

    logger.info(f"Output answers file {answers_path} already exists.")
    answers_obj = load_answers(answers_path, filter_empty_answers=True)
    answered_qids = answers_obj.unique_qids()

    df_queries = df_queries.copy()
    df_queries['qid'] = df_queries['qid'].astype(str)
    filtered = df_queries[~df_queries['qid'].isin(answered_qids)]

    if len(filtered) == 0:
        logger.info("No remaining queries to process. Exiting.")
    else:
        logger.info(f"Filtered queries to {len(filtered)} remaining queries to process.")
    return filtered

def process_queries_batch(queries_df: pd.DataFrame, agent, dataframe_extractor: DataFrameExtractor, output_paths, verbose: bool = False) -> int:
    """
    Process a batch of queries through the agent and extract the relevant information to save to output files.
    Returns the number of queries processed.

    Args:
        queries_df: DataFrame containing the queries to process. Must contain 'qid' and 'query' columns.
        agent: The agent to use for processing the queries. Must be callable with signature agent(queries_df) and return a DataFrame with the search results.
        dataframe_extractor: DataFrameExtractor to use for extracting the relevant information from the search results DataFrame.
        output_paths: OutputPaths object containing the paths to save the extracted information.
    """
    try:
        res = agent(queries_df)
    except Exception as e:
        raise RuntimeError(f"Error during agent execution: {e}. Queries: {queries_df}")
    try:
        dataframe_extractor.extract_all_and_save(res, output_paths, verbose=verbose)
    except e:
        pass


def load_queries(args) -> pd.DataFrame:
    """ Load queries from the specified dataset using the load_dataset function.
    Returns a DataFrame with 'qid' and 'query' columns."""
    dataset_loader = load_dataset
    df_queries = dataset_loader(dataset=args.dataset, queries=True, judged=args.judged_queries_only, qrels=False)
    return df_queries


def build_agent(args, pipeline):
    """
    Given args and a pyterrier retrieval pipeline, builds an agent.
    """
    from .rag_agent import build_agent
    agent = build_agent(model_type=args.model_type, model_id=args.model_id, retrieval_pipeline=pipeline, agent_k=args.search_r1_k)
    return agent

