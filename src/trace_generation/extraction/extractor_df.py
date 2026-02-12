import pandas as pd
from ..output_paths import OutputPaths

import enum
import os

def strip_text(query: str) -> str:
    if isinstance(query, str):
        return query.strip().replace('\n', ' ')
    return query


def save_or_append_df(entry_artifact_df: pd.DataFrame, fpath: str):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    if not os.path.exists(fpath):
        entry_artifact_df.to_csv(fpath, index=False, sep="\t", header=True, mode='w')
    else:
        entry_artifact_df.to_csv(fpath, index=False, sep="\t", header=False, mode='a')

class ExtractionType(enum.Enum):
    ANSWERS = "answers"
    THOUGHTS = "thoughts"
    ITER_QUERIES = "iter_queries"
    RETRIEVED_DOCS = "retrieved_docs"


def extract_answers(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df[["qid", "answer"]]


def extract_thoughts(results_df: pd.DataFrame) -> pd.DataFrame:
    df_exploded = results_df.explode('all_thoughts', ignore_index=True).dropna(subset=['all_thoughts'])
    df_exploded = df_exploded[['all_thoughts', 'qid']]

    query_parts = pd.DataFrame(
        df_exploded['all_thoughts'].tolist(), 
        index=df_exploded.index, 
        columns=['iteration', 'thought'] # separate it number and thought text
    )
    df_exploded = df_exploded.join(query_parts)
    df_exploded = df_exploded.drop(columns=['all_thoughts'])
    
    df_exploded['thought'] = df_exploded['thought'].apply(strip_text)
    return df_exploded


def extract_iter_queries(results_df: pd.DataFrame) -> pd.DataFrame:
    df_exploded = results_df.explode("all_queries", ignore_index=True).dropna(subset=["all_queries"])
    df_exploded = df_exploded[["all_queries", "qid"]]

    query_parts = pd.DataFrame(
        df_exploded["all_queries"].tolist(),
        index=df_exploded.index,
        columns=["iteration", "llm_query"],
    )
    df_exploded = df_exploded.join(query_parts)
    df_exploded = df_exploded.drop(columns=["all_queries"])

    df_exploded["llm_query"] = df_exploded["llm_query"].apply(strip_text)
    return df_exploded


def extract_retrieved_docs(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        raise ValueError("Ranked List is empty.")

    empty_list_docids = results_df[results_df["all_docids"].apply(lambda x: x is None or len(x) == 0)]
    if len(empty_list_docids) == len(results_df):
        """
            NOTE: change this if need to support >1 query
        """
        raise ValueError("No retrieved documents found in ranked list DataFrame.")

    # explode outer list (it, list retrieved docs)
    df_exploded = results_df.explode("all_docids", ignore_index=True).dropna(subset=["all_docids"])
    df_exploded = df_exploded[["all_docids", "qid"]]

    # split into two cols: it number and list of retrieved docids
    docs_parts = pd.DataFrame(
        df_exploded["all_docids"].tolist(),
        index=df_exploded.index,
        columns=["iteration", "retrieved_docs"], # separate it number and retrieved docs
    )
    df_exploded = df_exploded.join(docs_parts)
    df_exploded = df_exploded.drop(columns=["all_docids"])

    df_exploded = df_exploded.explode("retrieved_docs", ignore_index=False).dropna(subset=["retrieved_docs"])
    df_exploded = df_exploded.rename(columns={"retrieved_docs": "docid"})
    df_exploded["rank"] = df_exploded.groupby(["qid", "iteration"]).cumcount() + 1

    df_exploded = df_exploded.dropna(subset=["docid"])
    df_exploded["rank"] = df_exploded["rank"].astype("Int64")
    df_exploded["docid"] = df_exploded["docid"].astype("Int64")
    return df_exploded

def extract_save_wrapper(extraction_fn, df, fpath):
    res = extraction_fn(df)
    save_or_append_df(entry_artifact_df=res, fpath=fpath)


EXTRACTION_FUNCTIONS = {
    ExtractionType.ANSWERS: lambda df, output_paths: extract_save_wrapper(extraction_fn=extract_answers, df=df, fpath=output_paths.answers),
    ExtractionType.THOUGHTS: lambda df, output_paths: extract_save_wrapper(extraction_fn=extract_thoughts, df=df, fpath=output_paths.thoughts),
    ExtractionType.ITER_QUERIES: lambda df, output_paths: extract_save_wrapper(extraction_fn=extract_iter_queries, df=df, fpath=output_paths.iter_queries),
    ExtractionType.RETRIEVED_DOCS: lambda df, output_paths: extract_save_wrapper(extraction_fn=extract_retrieved_docs, df=df, fpath=output_paths.retrieved_docs),
}


class DataFrameExtractor():

    def __init__(self, extraction_order: list[ExtractionType] = None, verbose: bool = False):
        if extraction_order is None:
            extraction_order = [ExtractionType.ANSWERS, ExtractionType.THOUGHTS, ExtractionType.ITER_QUERIES, ExtractionType.RETRIEVED_DOCS]
        self.extraction_order = extraction_order
        self.verbose = verbose
        if self.verbose:
            print(f"Initialized DataFrameExtractor with extraction order: {[et.value for et in self.extraction_order]}")
    
    def _parse_extraction_fn(self, extraction_type_str: str) -> callable:
        extracttion_fn = EXTRACTION_FUNCTIONS.get(ExtractionType(extraction_type_str))
        if extracttion_fn is None:
            raise ValueError(f"Invalid extraction type: {extraction_type_str}. Valid types are: {[et.value for et in ExtractionType]}")
        return extracttion_fn
    
        
    def _extract_and_save(self, extraction_type: ExtractionType, results_df: pd.DataFrame, output_paths: OutputPaths) -> bool:
        extraction_func = self._parse_extraction_fn(extraction_type.value)
        df_extracted = extraction_func(results_df, output_paths)
        # TODO: save df
        
        return df_extracted
    
    def _prepare_df(self, results_df: pd.DataFrame, parse_thoughts: bool) -> pd.DataFrame:
        cols = ['qid', 'query', 'answer', 'iteration', 'all_queries', "all_docids"]
        if parse_thoughts:
            cols.append("all_thoughts")
        res = results_df[cols]
        return res
    
       
    def extract_all_and_save(self, results_df: pd.DataFrame, output_paths: OutputPaths, verbose: bool = False) -> dict[ExtractionType, pd.DataFrame]:
        res_df = self._prepare_df(results_df, parse_thoughts=(ExtractionType.THOUGHTS in self.extraction_order))
        extracted_dfs = {}
        for extraction_type in self.extraction_order:
            if self.verbose:
                print(f"Extracting {extraction_type.value}...")
            try:
                df_extracted = self._extract_and_save(extraction_type, res_df, output_paths)
                extracted_dfs[extraction_type] = df_extracted
            except Exception as e:
                if verbose:
                    print(f"Error extracting {extraction_type.value}: {e}")
        return extracted_dfs