import pandas as pd
from .researchy import ResearchyQuestionsDataset
import pyterrier as pt

from dataclasses import dataclass

@dataclass
class QuerySet:
    """Class representing a set of queries from a dataset, along with optional qrels."""
    dataset: str
    queries: pd.DataFrame
    qrels: pd.DataFrame = None

    def __repr__(self):
        return f"QuerySet(dataset={self.dataset}, num_queries={len(self.queries)})"
    
    def load(self, judged: bool = True, load_qrels: bool = False) -> None:
        """Load queries and optionally qrels for the dataset.
        Args:
            judged: whether to load only judged queries (if queries=True) or all queries. Ignored if queries=False.
            load_qrels: whether to load qrels instead of queries or corpus. If True, will load qrels. Ignored if queries=False.
        """
        self.queries = load_dataset(self.dataset, queries=True, judged=judged)

        assert "qid" in self.queries.columns, "Loaded queries must contain 'qid' column."
        assert "query" in self.queries.columns, "Loaded queries must contain 'query' column."
        self.queries['qid'] = self.queries['qid'].astype(str)

        if load_qrels:
            self.qrels = load_dataset(self.dataset, qrels=True)
            assert 'qid' in self.qrels.columns, "Loaded qrels must contain 'qid' column."
            assert 'docno' in self.qrels.columns, "Loaded qrels must contain 'docno' column."
            self.qrels['qid'] = self.qrels['qid'].astype(str)
            self.qrels['docno'] = self.qrels['docno'].astype(str)


def load_dataset(dataset: str, queries: bool = False, judged=True, qrels: bool = False)-> pd.DataFrame:
    """
    Load a dataset using the load_dataset function from the pyterrier.datasets.
    Supports also loading pyterrier datasets.

    dataset: name of the dataset to load. For researchy datasets, should be in the format 'researchy-{split}'.
    queries: whether to load queries or corpus. If False, will load corpus.
    judged: whether to load only judged queries (if queries=True) or all queries. Ignored if queries=False.
    qrels: whether to load qrels instead of queries or corpus. If True, will load qrels. Ignored if queries=False.
    """
    if "researchy" in dataset:
        split = dataset.split("_")[-1]
        dataset_obj = ResearchyQuestionsDataset(split=split, verbose=True, parse_qrels_flag=qrels)
        if qrels:
            return dataset_obj.load_qrels()
        return  dataset_obj.load_queries(judged=judged) if queries else dataset_obj.get_corpus_iter()

    if dataset == "msmarco-dev":
        dataset_irds = "irds:msmarco-passage/dev"
    elif dataset == "hotpotqa-test": 
        dataset_irds = "irds:beir/hotpotqa/test"
    else:
        dataset_irds = f"irds:{dataset}"

    try:
        dataset_obj = pt.datasets.get_dataset(dataset_irds)
    except Exception as e:
        raise ValueError(f"Error loading dataset {dataset_irds} from pyterrier: {e}")
    if qrels:
        return dataset_obj.get_qrels()
    return dataset_obj.get_topics() if queries else dataset_obj.get_corpus_iter()