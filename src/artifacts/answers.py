from typing import Iterable, List, Union
import pandas as pd
import os
from dataclasses import dataclass
from .artifacts import validate_and_load_artifact, get_artifact_schema

@dataclass
class AnswersArtifact:
    """
    Class representing the answers artifact: contains the answers for a set of queries.
    The answers are read from the TSV file and stored in a DataFrame with columns 'qid' and 'answer'.
    """
    answers_df: pd.DataFrame
    artifact_schema = get_artifact_schema("answers")


    def __post_init__(self):
        """
        Validate the answers DataFrame and set the index to 'qid'.
        """
        self.artifact_schema.validate_columns(self.answers_df)
        self.artifact_schema.enforce_dtypes(self.answers_df)
        self.answers_df = self.answers_df.set_index('qid', drop=False)

    def unique_qids(self) -> List[str]:
        """
        Return the list of unique qids in the answers DataFrame (as strings).
        """
        return self.answers_df['qid'].unique().tolist()
    
    def nunique_qids(self) -> int:
        """
        Return the number of unique qids in the answers DataFrame.
        """
        return self.answers_df['qid'].nunique()
    
    def num_answers(self) -> int:
        """
        Return the number of answers in the answers DataFrame.
        """
        return self.nunique_qids()
    
    def get_answer_for_qid(self, qid: str) -> str:
        """
        Returns the answer for the given qid.
        If multiple answers are found for the qid, returns the first one and prints a warning.
        """
        answers = self.answers_df.loc[qid, 'answer']
        if isinstance(answers, pd.Series) and len(answers) > 1:
            print("Warning: Multiple answers found for qid ", qid, ". Returning the first one.")
            return answers.iloc[0]
        return answers
    
    def exclude_qids(self, qids: Iterable[str]) -> 'AnswersArtifact':
        """
        Return a new AnswersArtifact that excludes the qids in the input iterable of qids.
        """
        qids = pd.Index(map(str, qids))
        remaining = self.answers_df.index.difference(qids)

        return AnswersArtifact(self.answers_df.loc[remaining])

    def subset_for_qids(self, qids: Iterable[str]) -> 'AnswersArtifact':
        """
        Return a new AnswersArtifact including only the provided qids.
        """
        qids = pd.Index(map(str, qids))
        return AnswersArtifact(self.answers_df.loc[qids])

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
    

def _check_answer_validity(answer: object) -> bool:
    """
    Check if the answer is a non-empty string.
    """
    if not isinstance(answer, str):
        return False
    return bool(answer.strip())


def load_answers(answers_fpath: Union[str, os.PathLike], verbose: bool = False, filter_empty_answers: bool = True) -> AnswersArtifact:
    """
        Load answers from a TSV file into an AnswersArtifact that encapsulates a DataFrame.
        The TSV file is expected to have columns 'qid' and 'answer'.
        If filter_empty_answers is True, will filter out answers that are empty or not valid strings.
        Note: duplicate qids will be dropped, keeping only the first occurrence.
    """
    answers_df = validate_and_load_artifact(answers_fpath, artifact_type="answers")
    answers_df['qid'] = answers_df[answers_df['qid'].notnull()]['qid'].astype(str)
    
    if verbose:
        print("Loaded %d answers from %s" % (len(answers_df), answers_fpath))
        print(f"Columns: {answers_df.columns.tolist()}")
        print(f"Unique qids: {answers_df['qid'].nunique()}")
        print(f"First few rows:\n{answers_df.head()}")

    if filter_empty_answers:
        answers_df = answers_df[answers_df['answer'].apply(_check_answer_validity)]
        
    answers_df = answers_df.drop_duplicates(subset=['qid'],  keep="first")
    if verbose and filter_empty_answers:
        print(f"Filtered answers: {len(answers_df)} out of {len(answers_df)}")
    answers_obj = AnswersArtifact(answers_df=answers_df)
    return answers_obj
