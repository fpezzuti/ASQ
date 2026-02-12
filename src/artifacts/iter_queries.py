from typing import List, Optional, Dict, Union
import pandas as pd
import os
from dataclasses import dataclass
from .artifacts import validate_and_load_artifact, get_artifact_schema


def trim_outlier_iterations(df_iter: pd.DataFrame, trim_outliers: Optional[float] = None) -> pd.Series:
    """
    Computes the probability of each iteration number.
    """
    if trim_outliers is None or trim_outliers is False:
        return df_iter
    tmp_df = df_iter.copy()
    n_distinct_queries = tmp_df["qid"].nunique()
   
    iter_counts = tmp_df.groupby("iteration")["qid"].nunique()
    probs = (iter_counts / n_distinct_queries)
    
    valid_iterations = probs[probs >= trim_outliers].index if trim_outliers is not None else probs.index
   
    filtered_df = tmp_df[tmp_df["iteration"].isin(valid_iterations)]
    return filtered_df


@dataclass
class SyntheticQueriesArtifact:
    """
    Class representing the iterative queries artifact: contains the queries generated at each iteration for a set of qids.
    The iterative queries are read from the TSV file and stored in a DataFrame with columns 'qid', 'iteration', and 'llm_query'.
    """
    iter_queries_df: pd.DataFrame
    artifact_schema = get_artifact_schema("iter_queries")

    def __post_init__(self):
        self.artifact_schema.validate_columns(self.iter_queries_df)
        self.artifact_schema.enforce_dtypes(self.iter_queries_df)
        self.iter_queries_df = self.iter_queries_df.set_index('qid', drop=False)

    def include_original_queries(self, queries_df: pd.DataFrame, verbose: bool = False) -> 'SyntheticQueriesArtifact':
        """
        Return a new SyntheticQueriesArtifact that includes the original queries from the input DataFrame.
         The input DataFrame should have columns 'qid' and 'query'.
         The original queries will be assigned iteration number 0 and the column 'query' will be renamed to 'llm_query' to be consistent with the iterative queries DataFrame.
         The resulting DataFrame will have the original queries followed by the iterative queries, sorted by qid and iteration number.
        """
        if verbose:
            print("Number of original queries to include:", len(queries_df))
            print("Number of iterative queries before include:", len(self.iter_queries_df))
        orig_queries = queries_df.copy()
        orig_queries['iteration'] = 0
        orig_queries.rename(columns={'query': 'llm_query'}, inplace=True)
        all_queries = pd.concat([orig_queries, self.iter_queries_df], ignore_index=True)
        if verbose:
            print("Number of original queries after include:", len(all_queries))
        return SyntheticQueriesArtifact(all_queries)

    def num_search_calls(self) -> int:
        """
        Return the number of search calls, which is the number of unique (qid, iteration) pairs in the iter_queries_df.
        This assumes that each (qid, iteration) pair corresponds to one search call.
        """
        return self.iter_queries_df[["qid", "iteration"]].drop_duplicates().shape[0]
    
    def unique_qids(self) -> List[str]:
        """
        Return the list of unique qids in the iter_queries_df (as strings).
        """
        return self.iter_queries_df['qid'].unique().tolist()
    
    def unique_iterations(self) -> List[int]:
        """
        Return the list of unique iterations in the iter_queries_df (as integers).
        """
        return self.iter_queries_df['iteration'].unique().tolist()
    
    def nunique_iterations(self) -> int:
        """
        Return the number of unique iterations in the iter_queries_df.
        """
        return self.iter_queries_df['iteration'].nunique()

    def nunique_qids(self) -> int:
        """
        Return the number of unique qids in the iter_queries_df.
        """
        return self.iter_queries_df['qid'].nunique()
    
    def nunique_queries(self) -> int:
        """
        Return the number of unique queries in the iter_queries_df.
        """
        return self.iter_queries_df['llm_query'].nunique()
    
    def max_iteration(self) -> int:
        """
        Maximum iteration number in the iter_queries_df.
        """ 
        return self.iter_queries_df['iteration'].max()
    
    def min_iteration(self) -> int:
        """
        Minimum iteration number in the iter_queries_df.
        """
        return self.iter_queries_df['iteration'].min()
    
    def mean_iteration(self) -> float:
        """
        Mean iteration number in the iter_queries_df.
        """
        return self.iter_queries_df['iteration'].mean()
    
    def std_iteration(self) -> float:
        """
        Standard deviation of iteration numbers in the iter_queries_df.
        """
        return self.iter_queries_df['iteration'].std()
    
    def median_iteration(self) -> float:
        """
        Median iteration number in the iter_queries_df.
        """
        return self.iter_queries_df['iteration'].median()
    
    def it_queries(self, iteration: int) -> 'SyntheticQueriesArtifact':
        """
        Return a new SyntheticQueriesArtifact containing only the queries for the specified iteration.
        """
        filtered_df = self.iter_queries_df[self.iter_queries_df['iteration'] == iteration]
        return SyntheticQueriesArtifact(filtered_df)
    
    def iteration_counts(self) -> pd.Series:
        """
        Return a Series with the counts of queries per iteration.
        """
        return self.iter_queries_df['iteration'].value_counts().sort_index()
    
    def query_trace_lengths(self) -> dict:
        """
        Return a dictionary mapping each qid to its maximum iteration number.
        """
        return (self.iter_queries_df.groupby(level=0)['iteration'].max().to_dict())
    
    def mean_query_length(self, mode: str = 'terms') -> float:
        """
        Compute the mean query length in terms or characters.
        mode: 'terms' or 'chars' to specify the unit of query length.
        """
        return self.query_length(mode=mode, aggregate=True)
    
    def std_query_length(self, mode: str = 'terms') -> float:
        """
        Compute the standard deviation of query length in terms or characters.
        mode: 'terms' or 'chars' to specify the unit of query length.
        Note: this computes the std of query lengths across all queries, not per iteration.
        """
        if mode == 'terms':
            len_fn = lambda x: len(x.split())
        elif mode == 'chars':
            len_fn = lambda x: len(x)
        else:
            raise ValueError(f"Unknown mode '{mode}' for std_query_length. Use 'terms' or 'chars'.")
           
        qlen_series = self.iter_queries_df['llm_query'].apply(len_fn)
        return qlen_series.std()
    
    def query_length(self, mode: str = 'terms', aggregate: bool = True) -> float:
        """
        Compute the mean query length in terms or characters.
        mode: 'terms' or 'chars' to specify the unit of query length.
        If aggregate is True, returns the mean query length across all queries. If False, returns a list of query lengths for each query.
        """
        if mode == 'terms':
            len_fn = lambda x: len(x.split())
        elif mode == 'chars':
            len_fn = lambda x: len(x)
        else:
            raise ValueError(f"Unknown mode '{mode}' for query_length. Use 'terms' or 'chars'.")
           
        qlen_series = self.iter_queries_df['llm_query'].apply(len_fn)
        return qlen_series.mean() if aggregate else qlen_series.tolist()

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
    

    def trim_outlier_iterations(self, trim_outliers: float) -> 'SyntheticQueriesArtifact':
        """
        Return a new SyntheticQueriesArtifact excluding iterations that are outliers based on the provided threshold.
        The <trim_outliers> parameter specifies the minimum proportion of queries that must have an iteration number for it to be included in the resulting artifact.
        """
        trimmed_df = trim_outlier_iterations(self.iter_queries_df, trim_outliers=trim_outliers)
        return SyntheticQueriesArtifact(trimmed_df)
    
    def assign_subquery_ids(self) -> 'SyntheticQueriesArtifact':
        """
        Return a new SyntheticQueriesArtifact with an additional column 'subquery_id' that assigns a unique ID to each (qid, iteration) pair.
        """
        from .traces import calc_subquery_id
        df_with_subqids = self.iter_queries_df.copy()
        df_with_subqids['subquery_id'] = df_with_subqids.apply(lambda row: calc_subquery_id(str(row['qid']), int(row['iteration'])),
                                                                axis=1)
        return SyntheticQueriesArtifact(df_with_subqids)
    
    def get_subqueries(self, qid: Optional[str] = None) -> Dict[str, str]:
        """
        Return a dictionary mapping subquery_id to llm_query for the given qid. If qid is None, return the mapping for all qids.
         Note: this requires that assign_subquery_ids() has been run first to create the 'subquery_id' column in the iter_queries_df.
        """
        if 'subquery_id' not in self.iter_queries_df.columns:
            raise ValueError("subquery_id column not found. Please run assign_subquery_ids() first.")
        if qid is not None:
            filtered_df = self.iter_queries_df[self.iter_queries_df['qid'] == str(qid)]
            subqueries_dict = dict(zip(filtered_df['subquery_id'], filtered_df['llm_query']))
            return subqueries_dict
        subqueries_dict = dict(zip(self.iter_queries_df['subquery_id'], self.iter_queries_df['llm_query']))
        return subqueries_dict
    
    
    
def load_iter_queries(iter_queries_fpath: Optional[Union[str, os.PathLike]], first_it_1: bool = True, verbose: bool = False) -> SyntheticQueriesArtifact:
    """
    Load the synthetic, iterative queries from a TSV file into a DataFrame.
    The TSV file is expected to have columns 'qid', 'iteration', and 'llm_query'.
    If first_it_1 is True, will change the iteration indexing from 0-based to 1-based by adding 1 to the iteration numbers.
    """
    df = validate_and_load_artifact(iter_queries_fpath, artifact_type="iter_queries")
   
    if verbose:
        print(f"Loaded {len(df)} iterative queries from {iter_queries_fpath}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Unique qids: {df['qid'].nunique()}")
        print(f"First few rows:\n{df.head()}")

    df_iter_no_dups = df.drop_duplicates(subset=['qid', 'iteration'])
    if verbose:
        print(f"After dropping duplicates: {len(df_iter_no_dups)} iter queries")
    df = df_iter_no_dups
    df['iteration'] = df['iteration'].astype(int)

    df = df[df['llm_query'].notnull()]
    df['llm_query'] = df['llm_query'].astype(str)
    if verbose:
        print(f"After removing empty query: {len(df)} iter queries")
    if first_it_1:
        df['iteration'] = df['iteration']+1 # change indexing from 0-based to 1-based
        if verbose:
            print("Changed iteration indexing from 0-based to 1-based.")
    return SyntheticQueriesArtifact(iter_queries_df=df)