from typing import Iterable, List, Optional, Union
import pandas as pd
import os
from dataclasses import dataclass
from .artifacts import validate_and_load_artifact, get_artifact_schema


@dataclass
class RetrievedDocsArtifact:
    """
    Class representing the retrieved documents artifact: contains the documents retrieved for each query and iteration.
    The retrieved documents are read from a DataFrame with columns 'qid', 'iteration', and 'docid'.
    """
    retrieved_docs_df: pd.DataFrame
    df_run: Optional[pd.DataFrame] = None # will be created by format_runs_into_trec_format() when needed
    artifact_schema = get_artifact_schema("retrieved_docs")

    def __post_init__(self):
        self.artifact_schema.validate_columns(self.retrieved_docs_df)
        self.artifact_schema.enforce_dtypes(self.retrieved_docs_df)

        
    def num_search_calls(self) -> int:
        """
        Return the number of search calls, which is the number of unique (qid, iteration) pairs in the retrieved_docs_df.
        """
        print("Warning: num_search_calls not implemented yet.")
        pass
    
    def unique_qids(self) -> List[str]:
        """
        Return the list of unique qids in the retrieved_docs_df (as strings).
        """
        return self.retrieved_docs_df['qid'].unique().tolist()
    
    def unique_iterations(self) -> List[int]:
        """
        Return the list of unique iterations in the retrieved_docs_df (as integers).
        """
        return self.retrieved_docs_df['iteration'].unique().tolist()
    
    def nunique_iterations(self) -> int:
        """
        Return the number of unique iterations in the retrieved_docs_df.
        """
        return self.retrieved_docs_df['iteration'].nunique()

    def nunique_qids(self) -> int:
        """
        Return the number of unique qids in the retrieved_docs_df.
        """
        return self.retrieved_docs_df['qid'].nunique()
    
    def subset_iterations(self, iterations: Iterable[int]) -> 'RetrievedDocsArtifact':
        """
        Return a new RetrievedDocsArtifact including only the provided iterations.
        """
        iterations = pd.Index(map(int, iterations))
        return RetrievedDocsArtifact(self.retrieved_docs_df[self.retrieved_docs_df["iteration"].isin(iterations)])
    
    def iteration_retrieved_docs(self, iteration: int) -> 'RetrievedDocsArtifact':
        """
        Return a RetrievedDocsArtifact of retrieved documents for a specific iteration.
        """
        return RetrievedDocsArtifact(self.retrieved_docs_df[self.retrieved_docs_df['iteration'] == iteration])
    
    def query_retrieved_docs(self, qid: str) -> 'RetrievedDocsArtifact':
        """
        Return a RetrievedDocsArtifact of retrieved documents for a specific qid.
        """
        retrieved_docs_df = self.retrieved_docs_df[self.retrieved_docs_df['qid'] == str(qid)]
        return RetrievedDocsArtifact(retrieved_docs_df=retrieved_docs_df)
    
    def docids_for_qid(self, qid: str) -> List[str]:
        """
        Return the list of docids retrieved for a specific qid.
        """
        try:
            docs = self.retrieved_docs_df[self.retrieved_docs_df['qid'] == str(qid)]['docid']
        except KeyError:
            return []

        if isinstance(docs, pd.Series):
            return docs.tolist()
        return [docs]
    
    def docid_sets_by_iteration(self) -> dict[int, dict[str, set]]:
        """
            Return a nested dictionary mapping iteration number to
            a dictionary mapping qid to the set of docids retrieved for that qid and iteration.
        Returns:
            { iteration -> { qid -> set(docids) } }
        """
        # group by iteration first, then qid
        grouped = self.retrieved_docs_df.groupby("iteration")
        result = {}
        for it, df_it in grouped:
            result[it] = df_it.groupby("qid")["docid"].apply(set).to_dict()
        return result
    
    def format_runs_into_trec_format(self, run_label: str) -> None:
        """
        Format the retrieved documents into TREC run format and store in self.df_run.
        The TREC run format has columns: 'qid', 'q0', 'docid', 'rank', 'score', 'run_id'.
         The 'qid' is formatted as "{original_qid}_iter={iteration}" to create a unique qid for each iteration of each original qid.
         The 'rank' is determined by the order of the retrieved documents for each (qid, iteration) pair in the retrieved_docs_df (starting from 1).
         The 'score' is set to 1/rank as a placeholder (since we don't have actual scores). The 'run_id' is set to the provided run_label.
         After this method is called, self.df_run will contain the formatted run DataFrame that can be used for evaluation with TREC tools.
        """
        from .traces import calc_subquery_id
        df_run = self.retrieved_docs_df.copy()
        df_run = df_run.sort_values(by=['qid', 'iteration'])
        df_run['qid'] = df_run.apply(lambda row: calc_subquery_id(str(row['qid']), int(row['iteration'])), axis=1)
        df_run['run_id'] = run_label
        df_run['q0'] = '0'
        df_run['score'] = 1/df_run['rank']
        df_run['docid'] = df_run['docid'].astype(str)
        df_run = df_run[['qid', 'q0', 'docid', 'rank', 'score', 'run_id']]
        self.df_run = df_run

    def get_it_run(self, iteration: int) -> pd.DataFrame:
        """
        Return the subset of the formatted run DataFrame (self.df_run) corresponding to the specified iteration.
        This method requires that format_runs_into_trec_format() has been called first to create self.df_run.
        The returned DataFrame will have the same columns as the TREC run format: 'qid', 'q0', 'docid', 'rank', 'score', 'run_id'.
        """
        if self.df_run is None:
            raise ValueError("df_run is not formatted yet. Call format_runs_into_trec_format() first.")
        return self.df_run[self.df_run['qid'].str.endswith(f"_iter={iteration}")]

    def helper(self) -> None:
        print(f"-------------------------------------[ {self.__class__.__name__} Helper ]-------------------------------------")
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
        print("\n-------------------------------------------------------------------------------------------------------------")


    
def load_retrieved_docs(retr_docs_fpath: Union[str, os.PathLike], first_it_1: bool = True, verbose: bool = False) -> RetrievedDocsArtifact:
    """
    Load retrieved documents from a TSV file into a DataFrame.
    The TSV file is expected to have columns 'qid', 'iteration', and 'docid'.
    If first_it_1 is True, will change the iteration indexing from 0-based to 1-based by adding 1 to the iteration numbers.
    """
    
    df = validate_and_load_artifact(file_path=retr_docs_fpath, artifact_type="retrieved_docs")

    if verbose:
        print(f"Loaded {len(df)} retrieved documents from {retr_docs_fpath}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Unique qids: {df['qid'].nunique()}")
        print(f"First few rows:\n{df.head()}")
    #df['iteration'] = df['iteration'].astype(int)
    if first_it_1:
        df['iteration'] = df['iteration']+1 # change indexing from 0-based to 1-based
    if verbose:
        print("Changed iteration indexing from 0-based to 1-based.")


    return RetrievedDocsArtifact(retrieved_docs_df=df)