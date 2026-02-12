import pandas as pd
import os
from typing import Tuple, Optional
from tqdm import tqdm
from typing import List
from .cw22b import get_corpus_iter, navigate_to_id


SUPPORTED_SPLITS = ['train', 'test', 'all']

DATA_DIR = "./../../data/" # TODO : move to config file

def download_researchy_questions(split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads the researchy questions dataset for the specified split (test or train) from HuggingFace.
    Returns a DataFrame with the questions and a DataFrame with the metadata.
    If split is "all", downloads train and test and concatenates them.
    """
    if split == "all":
        df_queries_train, df_metadata_train = download_researchy_questions("train")
        df_queries_test, df_metadata_test = download_researchy_questions("test")
        df_queries = pd.concat([df_queries_train, df_queries_test], ignore_index=True)
        df_metadata = pd.concat([df_metadata_train, df_metadata_test], ignore_index=True)
        return df_queries, df_metadata

    split_fname = "researchy_questions." + split + ".jsonl"
    df = pd.read_json("hf://datasets/corbyrosset/researchy_questions/" + split_fname, lines=True)
    df_queries = df[["id", "question"]]

    df_metadata = df[["id", "intrinsic_scores", "DocStream", "gpt4_decomposition", "decompositional_score", "nonfactoid_score"]]
    return df_queries, df_metadata

def extract_docstream_df(df_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the docstream information from the metadata DataFrame and return a DataFrame with columns 'qid', 'Url', 'Title', 'Snippet'.
    """
    df_qrels = df_metadata.copy()
    df_qrels = df_qrels[['id', 'DocStream']]
    df_qrels.rename(columns={'id': 'qid'}, inplace=True)
    df_qrels['DocStream'] = df_qrels['DocStream'].astype(object).to_list() # transform from str to list
    # DocStream can be a stringified list, a list, or missing.
    import ast
    def _parse_docstream(value):
        if isinstance(value, str):
            return ast.literal_eval(value)
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return []
    df_qrels['DocStream'] = df_qrels['DocStream'].map(_parse_docstream) # transform from str to list of dict
    df_exploded = df_qrels.explode('DocStream')

    # expand dictionary keys into columns
    df_final = pd.concat(
        [df_exploded.drop(columns=['DocStream']),
        df_exploded['DocStream'].apply(pd.Series)],
        axis=1
    )
    df_final['Url'] = df_final['Url'].astype(str).apply(lambda x: x.strip())
    return df_final

def parse_qrels(urls: List[str]) -> pd.DataFrame:
    """
    Parse the qrels information from the docstream and return a DataFrame with columns 'qid', 'docno', 'url', 'title'.
    """
    cw22b_metadata = []
    processed = 0
    
    for doc_data in tqdm(get_corpus_iter(start_txt_dir=f"{DATA_DIR}cw22b/txt/en/", ret_url=True)):
        if doc_data['url'].strip() in urls:
            doc_metadata = {
                'docno': doc_data['docno'],
                'url': doc_data['url'].strip(),
                'title': doc_data['title'].strip()}
            cw22b_metadata.append(doc_metadata)

        if processed % 1_000_000 == 0:
            print(f"Mapped {len(cw22b_metadata)} docnos to URLs.")
            print(f"Processed {processed} documents.")
        processed += 1

    df = pd.DataFrame(cw22b_metadata)
    return df


def queries_fpath(data_dir, dataset_id, split) -> str:
    return f"{data_dir}{dataset_id}_questions.{split}.tsv"

def qrels_fpath(data_dir, dataset_id, split) -> str:
    return f"{data_dir}{dataset_id}_questions.{split}.qrels.tsv"

def metadata_fpath(data_dir, dataset_id, split) -> str:
    return f"{data_dir}{dataset_id}_questions.{split}.metadata.tsv"

class ResearchyQuestionsDataset():
    dataset_id = "researchy"
    start_cw22b_dir = DATA_DIR + "/cw22b/txt/en/"

    def __init__(self, split: Optional[str] = None, verbose: bool = False, parse_qrels_flag: bool = False):
        """
        Initialize the ResearchyQuestionsDataset object.
        If the dataset files do not exist, will download and process the dataset.
        TODO: what split==None does?
        """
        if split is not None and split not in SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split {split} for ResearchyQuestionsDataset. \
                             Supported splits: {SUPPORTED_SPLITS}")
        self.split = split
        self.verbose = verbose
        self.data_dir = DATA_DIR + self.dataset_id + "/"

        os.makedirs(self.data_dir, exist_ok=True)

        if split is not None:
            splits_to_process = ['train', 'test'] if split == "all" else [split]

            for split in splits_to_process:
                questions_fpath = queries_fpath(self.data_dir, self.dataset_id, split)
                meta_fpath = metadata_fpath(self.data_dir, self.dataset_id, split)

                if not os.path.exists(questions_fpath):
                    if self.verbose:
                        print("Downloading Researchy questions dataset.")
                    df_questions, df_metadata = download_researchy_questions(split)

                    if self.verbose:
                        print(f"Loaded dataframe with length: {len(df_questions)} for split: {split}")
                    df_questions.to_csv(questions_fpath, index=False, sep='\t', header=False)

                    df_metadata.to_csv(meta_fpath, index=False, sep='\t', header=True) # save raw metadata

                    if self.verbose:
                        print(f"Saved metadata to {meta_fpath}")

                    if parse_qrels_flag:
                        qrels_file_path = qrels_fpath(self.data_dir, self.dataset_id, split)
                        if self.verbose:
                            print(f"Processing docstream for split: {split}.")
                        df_docstream = extract_docstream_df(df_metadata)
                        if self.verbose:
                            print("Number of docstream entries:", len(df_docstream))

                        urls = set(df_docstream['Url'].tolist())
                        if self.verbose:
                            print(f"Unique URLs: {len(urls)}")
                            print(f"Unique QIDs: {df_docstream['qid'].nunique()}")
                            print(f"Unique Titles: {df_docstream['Title'].nunique()}")
                            print(f"Unique Snippets: {df_docstream['Snippet'].nunique()}")
                            print(f"Total Rows: {len(df_docstream)}")
                            print("Duplicated URLs:", df_docstream.duplicated(subset=['qid', 'Url']).sum())

                        cw22b_metadata_df = parse_qrels(urls)

                        merged_df = pd.merge(df_docstream, cw22b_metadata_df, left_on='Url', right_on='url', how='left')

                        merged_df.head()
                        merged_df.drop(columns=['UrlLanguage', "CluewebURLHash"], inplace=True)

                        qrels_df_final = merged_df[['qid', 'docno', 'Snippet']]
                        qrels_df_final.dropna(subset=['docno'], inplace=True)

                        if self.verbose:
                            print(qrels_df_final.head())
                        qrels_df_final.to_csv(qrels_file_path, sep='\t', index=False, header=True)
                        if self.verbose:
                            print("Len(qrels_df_final): ", len(qrels_df_final))

                        judged_queries = set(qrels_df_final['qid'].tolist())
                        if self.verbose:
                            print(f"Number of judged queries: {len(judged_queries)}")
                            print(f"Avg number of judged docs per query: {len(qrels_df_final)/len(judged_queries):.2f}")
                            print(f"Distinct docs: {qrels_df_final['docno'].nunique()}")
                            print(f"Max docs per query: {qrels_df_final.groupby('qid').size().max()}")

                    if self.verbose:
                        print(f"Saved queryset info.")
                else:
                    if self.verbose:
                        print(f"Queries file {questions_fpath} already exists. Skipping download.")


    def get_corpus_iter(self):
        return get_corpus_iter(start_txt_dir=self.start_cw22b_dir, verbose=self.verbose, cat_docids=True, ret_url=False)

    def load_queries(self, judged: bool = False) -> pd.DataFrame:
        """
        Load the questions for the specified split.
        If judged is True, will only load the questions that have judged qrels.
        Returns a DataFrame with columns 'qid' and 'query' (like as done in pyterrier datasets).
        """ 
        splits_to_load = ['train', 'test'] if self.split == 'all' else [self.split]

        df_concatenated = pd.DataFrame()
        for split in splits_to_load:
            questions_fpath = queries_fpath(self.data_dir, self.dataset_id, split)
            df_queries = pd.read_csv(questions_fpath, sep='\t', header=None, names=['qid', 'query'])
            df_queries['qid'] = df_queries['qid'].astype(str)
            if judged:
                df_qrels = self.load_qrels(split=split, ret_snippet=False)
                judged_qids = set(df_qrels['qid'].tolist())
                df_queries = df_queries[df_queries['qid'].isin(judged_qids)].reset_index(drop=True)
            df_concatenated = pd.concat([df_concatenated, df_queries], ignore_index=True)
        return df_concatenated
    
    def load_qrels(self, split: str, ret_snippet: bool = False) -> pd.DataFrame:
        """
        Load the qrels for the specified split.
        If ret_snippet is True, will include the snippet column.
        Returns a DataFrame with columns 'qid', 'docno', 'Snippet' (if ret_snippet is True), and 'label'.
        """
        qrels_fpath = qrels_fpath(self.data_dir, self.dataset_id, split)
        df_qrels = pd.read_csv(qrels_fpath, sep='\t', header=0)

        if not ret_snippet:
            df_qrels = df_qrels.drop(columns=['Snippet'])
        
        df_qrels['label'] = 1  # all relevance labels are 1
        df_qrels['qid'] = df_qrels['qid'].astype(str)
        df_qrels['docno'] = df_qrels['docno'].astype(str)

        return df_qrels

    def get_doc_metadata(self, docno: str) -> dict:
        """
        Retrieve document metadata for a given docno (docno is a 11-digit identifier).
        """
        doc_data = navigate_to_id(dir_path=self.start_cw22b_dir, docid=docno, ret='all', cat_docid=True)
        return doc_data