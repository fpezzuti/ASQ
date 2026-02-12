import logging
import os
from typing import Optional

import pandas as pd
import pyterrier as pt
from pyterrier_dr import ElectraScorer

from common.config import DATA_DIR
from .researchy import ResearchyQuestionsDataset
from .load_mydataset import load_dataset

logger = logging.getLogger("gen_search_r1")


def calc_index_fpath(dataset: str) -> str:
    """
    Calculate the file path for the index of a given dataset.
    """
    if dataset.startswith("researchy"):
        return f"{DATA_DIR}/index/cw22b_bm25_index/"
    elif dataset == "hotpotqa-test":
        return f"{DATA_DIR}/index/hotpotqa_bm25_index/"
    elif dataset == "msmarco-passage":
        return f"{DATA_DIR}/index/msmarco_passage_bm25_index/"
    else:
        return f"{DATA_DIR}/index/{dataset}_bm25_index/"


def index_lexical(index_fpath: str, dataset: str, nthreads: int = 1) -> None:
    """
    Builds the lexical index for a given dataset.
    """
    if not os.path.exists(index_fpath):
        os.makedirs(index_fpath, exist_ok=True)
        logger.info(f"Created index directory at {index_fpath}.")
    else:
        logger.info(f"Index directory {index_fpath} already exists.")
        return
    
    if "msmarco" in dataset:
        meta = {"docno": 20, "text": 512}
    elif "cw22b" in index_fpath:
        meta = {"docno": 20, "title": 50}
    elif "hotpot" in dataset:
        meta = {"docno": 20, "title": 50, "text": 512}
    else:
        meta = {"docno": 20, "text": 512}
        
    meta_reverse = ["docno"]
    properties = {
        "index.direct.enabled": "false",
        "termpipelines": "Stopwords,PorterStemmer",
        "ignore.empty.documents": "true",
    }
    indexer = pt.terrier.IterDictIndexer(
        index_fpath,
        blocks=False,
        meta_reverse=meta_reverse,
        meta=meta,
        type=pt.terrier.IndexingType.SINGLEPASS,
        stopwords="terrier",
        properties=properties,
        threads=nthreads,
    )
    logger.info("Loading corpus for indexing.")
    corpus_iter = load_dataset(dataset, queries=False) # load corpus
    logger.info("Starting indexing.")
    indexer.index(corpus_iter) # index
    logger.info("Indexing completed.")


class CW22BLexicalRetriever(pt.Transformer):
    """
    Custom retriever for the CW22B dataset that retrieves the top-k documents for each query and also retrieves the document text and title for each retrieved document.
    Metadata are retrieved separately for efficiency reasons.
    """
    def __init__(self, lexical_index_path: str, topk: int, wmodel: str = "BM25", truncate: Optional[int] = None):
        self.retriever = pt.terrier.Retriever(lexical_index_path, wmodel=wmodel, metadata=["docno", "title"])
        self.topk = topk
        logger.info("CW22BLexicalRetriever initialized.")
        self.dataset_obj = ResearchyQuestionsDataset(split=None, verbose=True, parse_qrels_flag=False)
        logger.info("ResearchyQuestionsDataset initialized for document retrieval.")
        self.truncate = truncate

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """
        Retrieve the top-k documents for each query and also retrieve the document text and title for each retrieved document.
        The input DataFrame must contain a 'qid' column with the query ids and a 'query' column with the query text.
        Returns a DataFrame with columns 'qid', 'docno', 'title', 'text', and 'rank' (the rank of the retrieved document for the query). The 'text' column is truncated to the specified number of characters if truncate is not
        """
        results = self.retriever.transform(inp)
        docids = results["docno"]

        for docid in list(docids.unique()):
            doc_data = self.dataset_obj.get_doc_metadata(docno=docid)
            doc_text = doc_data["text"]
            if self.truncate is not None:
                doc_text = doc_text[: self.truncate]
            title = doc_data["title"]
            results.loc[results["docno"] == docid, "title"] = title
            results.loc[results["docno"] == docid, "text"] = doc_text

        sorted_results = results.sort_values(by=["qid", "rank"]).reset_index(drop=True)
        results = sorted_results
        results_topk = results.groupby("qid").head(self.topk).reset_index(drop=True)
        return results_topk


def build_retrieval_pipeline(dataset: str, k1: int = 1000, k2: Optional[int] = None, batch_size: int = 128, wmodel: str = "BM25") -> pt.Transformer:
    """
    Build the retrieval pipeline for a given dataset.
    The pipeline consists of a lexical retriever (BM25) that retrieves the top-k1 documents for each query,
    and optionally an Electra reranker that reranks the retrieved documents and returns the top-k2 documents for each query (if k2 is not None).
    Returns the retrieval pipeline as a pyterrier Transformer.
    """
    lexical_index_path = calc_index_fpath(dataset)
    index_lexical(lexical_index_path, dataset)

    logger.info(f"Directory {lexical_index_path} exists: {os.path.exists(lexical_index_path)}")
    if "cw22" in lexical_index_path:
        retriever = CW22BLexicalRetriever(lexical_index_path, topk=k1, wmodel=wmodel, truncate=512)
        pipeline = retriever
        return pipeline
    retriever = pt.terrier.Retriever(lexical_index_path, wmodel=wmodel, metadata=["docno", "text"])
    if k2 is not None:
        electra_reranker = ElectraScorer(batch_size=batch_size, verbose=False, device="cuda")
        pipeline = retriever % k1 >> electra_reranker % k2
        logger.info(f"Retrieval pipeline created (k1={k1}, k2={k2}).")
        return pipeline

    pipeline = retriever % k1
    logger.info("Retrieval pipeline created (k1=%s, no reranking).", k1)
    return pipeline