from dataclasses import dataclass
import pandas as pd
from typing import Optional, Tuple
from pathlib import Path


from trace_analysis.eval_utils.eval_utils import print_stats
from data_utils.load_mydataset import load_dataset
from trace_generation.output_paths import OutputPaths, build_output_paths
from huggingface_hub import HfApi
from .artifacts import download_artifact
from .answers import AnswersArtifact, load_answers
from .iter_queries import SyntheticQueriesArtifact, load_iter_queries
from .retrieved_docs import RetrievedDocsArtifact, load_retrieved_docs

REPOSITORY_ID = "AgenticSearchQueryset/ASQ" 

def calc_subquery_id(qid: str, iteration: int) -> str:
    """
    Calculate a subquery ID based on the original qid and iteration number.
    """
    return f"{qid}_iter={iteration}"

def calc_original_query_id(subquery_id: str) -> str:
    """
    Parse the original qid from a subquery qid.
    """
    if "_iter=" in subquery_id:
        return subquery_id.split("_iter=")[0]
    return subquery_id

def extract_it_from_subquery_id(subquery_id: str) -> int:
    """
    Parse the iteration number from a subquery qid.
    """
    """Parse the iteration number from a subquery qid."""
    if "_iter=" in subquery_id:
        return int(subquery_id.split("_iter=")[1])
    raise ValueError(f"subquery_id '{subquery_id}' does not contain iteration info.")

@dataclass
class TraceCollectionConfig:
    """
    Configuration for a trace collection.
    
    Fields:
      dataset: queryset name.
      data_dir: directory where traces' artifacts are stored.
      retriever: retriever name.
      retriever_k: retriever cut-off.
      agent_family: family of the agent (e.g. search_r1).
      agent_generator: generator of the agent (e.g., Qwen-7B).
      reranker: reranker name (optional).
      reranker_k: reranker cut-off(optional).
    """
    dataset: str
    data_dir: str
    retriever: str
    retriever_k: int
    agent_family: str
    agent_generator: str
    reranker: Optional[str] = None
    reranker_k: Optional[int] = None

    def __repr__(self):
        return (
            "TraceCollectionConfig("
            f"dataset={self.dataset}, retriever={self.retriever}, retriever_k={self.retriever_k}, "
            f"reranker={self.reranker}, reranker_k={self.reranker_k}, "
            f"agent_family={self.agent_family}, agent_generator={self.agent_generator})"
        )

    def build_output_paths(self) -> OutputPaths:
        """
        Build the output paths for the trace artifacts based on the configuration.
        The paths are organized as follows:
        <data_dir>/traces/<dataset>/<retriever>_k<retriever_k>_<reranker>_k<reranker_k>/<agent_family>/<agent_generator>/answers.tsv
        <data_dir>/traces/<dataset>/<retriever>_k<retriever_k>_<reranker>_k<reranker_k>/<agent_family>/<agent_generator>/iter_queries.tsv
        <data_dir>/traces/<dataset>/<retriever>_k<retriever_k>_<reranker>_k<reranker_k>/<agent_family>/<agent_generator>/retrieved_docs.tsv
        """
        return build_output_paths(
            data_dir=self.data_dir,
            dataset=self.dataset,
            retriever=self.retriever,
            retriever_k=self.retriever_k,
            reranker=self.reranker,
            reranker_k=self.reranker_k,
            agent_family=self.agent_family,
            agent_generator=self.agent_generator,
        )
    

@dataclass
class TraceCollection:
    """
    class for loading and manipulating the trace data for a given model and dataset. The trace data includes the original queries, the answers,
    the iterative queries, and the retrieved documents for each iteration.
    """
    config: TraceCollectionConfig
    queries_df: Optional[pd.DataFrame] = None
    answers: Optional[AnswersArtifact] = None
    iter_queries: Optional[SyntheticQueriesArtifact] = None
    retrieved_docs: Optional[RetrievedDocsArtifact] = None
    first_it_1: bool = True # whether the first iteration of the iterative queries is iteration 1 (instead of 0, which is the original query)

    def __repr__(self):
        config_str = repr(self.config)
        return f"TraceCollection(config={config_str})"
    
    def download_trace(self, repo_id: str, output_paths: OutputPaths, verbose: bool = False) -> None:
        """
        Download the trace artifacts from the Hugging Face repository with ID <repo_id> and load them into the TraceCollection instance.
        """
        self.queries_df = load_dataset(self.config.dataset, queries=True, judged=False, qrels=False)

        # infer subset path 
        base_parts = list(Path(output_paths.base_dir).parts)

        if "traces" in base_parts:
            subset = "/".join(base_parts[base_parts.index("traces"):])
        else:
            retriever_cfg = f"{self.config.retriever}_k{self.config.retriever_k}"

            if self.config.reranker and self.config.reranker_k:
                retriever_cfg = f"{retriever_cfg}_{self.config.reranker}_k{self.config.reranker_k}"
            subset = "/".join(["traces",
                               self.config.dataset,
                               retriever_cfg,
                               self.config.agent_family,
                               self.config.agent_generator])

        if verbose:
            print(f"Downloading trace artifacts from HF repo '{repo_id}' subset '{subset}'.")

        api = HfApi(token=True)
        output_paths.base_dir.mkdir(parents=True, exist_ok=True)

        _ = download_artifact(api=api, repo_id=repo_id, artifact_type="answers", subset=subset, filename=output_paths.answers.name, save_fpath=output_paths.answers)
        self.answers = load_answers(output_paths.answers, filter_empty_answers=True, verbose=verbose)

        _ = download_artifact(api=api, repo_id=repo_id, artifact_type="iter_queries", subset=subset, filename=output_paths.iter_queries.name, save_fpath=output_paths.iter_queries)
        self.iter_queries = load_iter_queries(iter_queries_fpath=output_paths.iter_queries, first_it_1=self.first_it_1, verbose=verbose)

        _ = download_artifact(api=api, repo_id=repo_id, artifact_type="retrieved_docs", subset=subset, filename=output_paths.retrieved_docs.name, save_fpath=output_paths.retrieved_docs)
        self.retrieved_docs = load_retrieved_docs(output_paths.retrieved_docs, verbose=verbose, first_it_1=self.first_it_1)
       

    def _trim_outlier_traces(self, trim_outliers: float, verbose: bool = False) -> None:
        self.iter_queries = self.iter_queries.trim_outlier_iterations(trim_outliers=trim_outliers)
        if verbose:
            print_stats(self.iter_queries.iter_queries_df)

    def _trim_outlier_traces_from_retrieved_docs(self, verbose: bool = False) -> None:
        """
        Remove the retrieved documents that correspond to the iterations that were trimmed as outliers in the iterative queries.
        This is to ensure consistency between the iterative queries and the retrieved documents.
        This should be called after _trim_outlier_traces.
        """
        iterations_to_keep = self.iter_queries.unique_iterations()
        self.retrieved_docs = self.retrieved_docs.subset_iterations(iterations_to_keep)
        if verbose:
            print_stats(self.retrieved_docs.retrieved_docs_df)

    def read_trace_artifacts_from_files(self, output_paths: OutputPaths, load_docs: bool = False, filter_empty_answers: bool = True, verbose: bool = False) -> None:
        """
        Read the trace artifacts from local files and load them into the TraceCollection instance.
        output_paths: OutputPaths object containing the paths to the trace artifacts.
        load_docs: whether to load the retrieved documents artifact.
        filter_empty_answers: whether to filter out empty answers
        verbose: whether to print verbose logs.
        """
        self.queries_df = load_dataset(self.config.dataset, queries=True, judged=False, qrels=False)

        self.answers: AnswersArtifact = load_answers(output_paths.answers, filter_empty_answers=filter_empty_answers, verbose=verbose)
        self.iter_queries: SyntheticQueriesArtifact = load_iter_queries(iter_queries_fpath=output_paths.iter_queries, first_it_1=self.first_it_1, verbose=verbose)

        if load_docs:
            self.retrieved_docs = load_retrieved_docs(output_paths.retrieved_docs, verbose=verbose, first_it_1=self.first_it_1)
    
    def load_data(self, load_docs: bool = False, trim_outliers: Optional[float] = None, filter_empty_answers: bool = True, verbose: bool = False) -> None:
        """
        Load the trace data (queries, answers, iterative queries, and optionally retrieved documents) from the output files based on the configuration.
        If trim_outliers is not None, will trim iterations that are outliers in terms of trace length.
        If filter_empty_answers is True, will filter out answers that are empty strings or non-string values.
        """
        output_paths = self.config.build_output_paths()

        try:
            print("Trying to read trace artifacts from local files.")
            self.read_trace_artifacts_from_files(output_paths=output_paths, load_docs=load_docs, filter_empty_answers=filter_empty_answers, verbose=verbose)
        except (OSError) as e:
            try: 
                if verbose:
                    print(f"Trace artifacts not found in local path. Attempting to download from Hugging Face repository {REPOSITORY_ID}.")
                self.download_trace(repo_id=REPOSITORY_ID, output_paths=output_paths, verbose=verbose)
            except Exception as download_exception:
                print("Error: Trace artifacts not found in local path and failed to download from Hugging Face repository.")
                print(f"Got exception: {e}.")
                print(f"Then got exception: {download_exception}")
                raise e
            
        if verbose:
            print_stats(self.iter_queries.iter_queries_df)
        
        if trim_outliers is not None:
            self._trim_outlier_traces(trim_outliers=trim_outliers, verbose=verbose)
            self._trim_outlier_traces_from_retrieved_docs(verbose=verbose)

        
    def directly_answered_qids(self) -> set[str]:
        """
        Return the set of qids that were directly answered (i.e., qids that are in the answers, but have no synthetic queries).
        """
        return set(self.answers.unique_qids()) - set(self.iter_queries.unique_qids())
    
    def failed_traces(self) -> set[str]:
        """
        Return the set of qids that failed to be answered (i.e., qids that are not in the answers).
        """
        orig_qids = self.original_qids()
        answered_qids = set(self.answers.unique_qids())
        return orig_qids - answered_qids
    
    def original_qids(self) -> set[str]:
        """
        Return the set of original qids from the dataframe of input queries.
        """
        return set(self.queries_df['qid'])
    
    def directly_failed_qids(self) -> set[str]:
        """
        Return the set of qids that failed to be answered (i.e., qids that are not in the answers and have involved no search calls).
        """
        orig_qids = self.original_qids()
        failed_traces = orig_qids - set(self.answers.unique_qids())
        directly_failed = failed_traces - set(self.iter_queries.unique_qids())
        return directly_failed
    
    def parse_iteration_probs(self, include_orig_query: bool = True, aggregate: bool = True) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the likelihood of each iteration of being the last one in a trace (i.e., the iteration at which the agent produced the final answer).
        If include_orig_query is True, the original queries (iteration 0) are included in the distribution.
        If aggregate is True, the probabilities are aggregated over all qids."""

        if not aggregate:
            raise NotImplementedError("Non-aggregate mode is not implemented for parse_iteration_probs.")
        traces_lengths_dict = self.iter_queries.query_trace_lengths()

        max_iter_distinct_values = sorted(list(set(traces_lengths_dict.values())))
        max_iter_counts = [list(traces_lengths_dict.values()).count(value) for value in max_iter_distinct_values]

        if include_orig_query: # include original queries as iteration 0   
            iterated_qids = self.iter_queries.unique_qids()
            one_shot_answers = self.answers.exclude_qids(qids=iterated_qids)
            max_iter_distinct_values = [0] + max_iter_distinct_values
            max_iter_counts = [one_shot_answers.nunique_qids()] + max_iter_counts

        counts = pd.Series(data=max_iter_counts, index=max_iter_distinct_values)
        probs = counts / counts.sum()
        return list(probs.index), list(probs.values)

    def helper(self) -> None:
        """
        Print a helper message describing the TraceCollection class and its methods.
        """
        print(f"----------------------------------------[ {self.__class__.__name__} Helper ]-------------------------------------------------")
        print(f"The {self.__class__.__name__} class exposes the following methods:\n")
        public_methods = [
            name for name in dir(self)
            if callable(getattr(self, name)) and not name.startswith("_")
        ]
        for idx, name in enumerate(sorted(public_methods), start=1):
            method = getattr(self, name)
            doc = (method.__doc__ or "").strip().splitlines()[0] if method.__doc__ else ""
            suffix = f": {doc}" if doc else ""
            print(f"{idx}. {name}(){suffix}")
        print("\n-------------------------------------------------------------------------------------------------------------------")
