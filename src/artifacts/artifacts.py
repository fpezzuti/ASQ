
from dataclasses import dataclass
import pandas as pd
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union

import io
import os
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download

TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool
}

def enforce_dtypes(df: pd.DataFrame, dtypes: Dict[str, type], artifact_type: str) -> pd.DataFrame:
    """
    Enforce specified dtypes on DataFrame columns (in-place).
    Raises a ValueError if any column cannot be converted to the specified dtype.
    """
    for col, dtype in dtypes.items():
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            raise ValueError(f"Failed to convert column '{col}' to {dtype} in artifact '{artifact_type}': {e}")
    return df


@dataclass
class ArtifactSchema:
    """
    Class representing the schema for an artifact type, including its name,
    allowed file extensions, required columns, and expected dtypes for the columns.
    """
    name: str
    extensions: List[str]
    required_columns: List[str]
    dtypes: Dict[str, type]

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
            Ensures all required columns are present in the DataFrame.
            Raises a ValueError if any required columns are missing.
        """
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"{self.name} missing columns: {missing_cols}. Found: {df.columns.to_list()}")
        return True
    
    def enforce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
            Enforce the expected dtypes on the DataFrame columns.
            Raises a ValueError if any column cannot be converted to the specified dtype.
        """
        return enforce_dtypes(df=df, dtypes=self.dtypes, artifact_type=self.name)
    

def load_artifact_registry(config_dir: str, verbose: bool = False) -> Dict[str, ArtifactSchema]:
    """ 
        Dynamically builds the registry from a directory of YAML files.
        Each YAML file should define an artifact schema with fields: name, extensions, required_columns, and dtypes.
    """
    registry = {}
    config_path = Path(config_dir)

    for yaml_file in config_path.glob("*.yaml"):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            if verbose:
                print(f"Loading artifact schema from {yaml_file}: {data}")
            
            if "dtypes" in data:
                data["dtypes"] = {k: TYPE_MAP.get(v, str) for k, v in data["dtypes"].items()}
            
            schema = ArtifactSchema(**data)
            registry[schema.name] = schema
            
    if verbose:
        print(f"Loaded {len(registry)} artifact schemas from {config_dir}")
    return registry


_CONFIG_DIR = Path(__file__).resolve().parent / "config" # initialize the registry
ARTIFACT_REGISTRY = load_artifact_registry(str(_CONFIG_DIR), verbose=False)

def get_artifact_schema(artifact_type: str) -> ArtifactSchema:
    """
    Get the artifact schema for the specified artifact type from the registry.
    Raises a ValueError if the artifact type is not found in the registry.
    """
    config = ARTIFACT_REGISTRY.get(artifact_type)
    if not config:
        raise ValueError(f"Artifact type '{artifact_type}' not found in registry. Available types: {list(ARTIFACT_REGISTRY.keys())}")
    return config

def df_to_bytesio(df: pd.DataFrame) -> io.BytesIO:
    """
    Convert a DataFrame to an in-memory binary TSV.
    The DataFrame is first converted to a TSV string, then encoded to bytes and written to a BytesIO object.
    """
    df = df.dropna(how="all")
    buffer = io.BytesIO()
    buffer.write(df.to_csv(index=False, sep="\t", encoding="utf-8", lineterminator="\n").encode("utf-8"))
    buffer.seek(0)
    return buffer



def validate_and_load_artifact(file_path: Union[str, os.PathLike], artifact_type: str) -> pd.DataFrame:
    """
    Validate the artifact file against the schema and load it into a DataFrame.
    """
    artifact_config: ArtifactSchema = ARTIFACT_REGISTRY.get(artifact_type)
    if not artifact_config:
        raise ValueError(f"Unknown artifact type '{artifact_type}'")
    
    file_path = os.fspath(file_path)
    _, ext = os.path.splitext(file_path) # get file extension
    ext = ext.lower()
    if ext not in artifact_config.extensions:
        raise ValueError(f"Invalid extension {ext}. Expected: {artifact_config.extensions}")

    df = None
    try:
        if ext == ".csv":
            df = pd.read_csv(file_path, sep=",")
        if ext == ".tsv":
            df = pd.read_csv(file_path, sep="\t")
        elif ext == ".json":
            df = pd.read_json(file_path)
        elif ext == ".jsonl":
            df = pd.read_json(file_path, lines=True)
        elif ext == ".txt":
            df = pd.read_csv(file_path, sep="\t", names=artifact_config.required_columns) # assume a single column
        elif ext == ".md":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
    except OSError as e:
        raise e
    except Exception as e:
        raise ValueError(f"Failed to parse {file_path}: {e}")

    artifact_config.validate_columns(df)
    artifact_config.enforce_dtypes(df)
    return df


def publish_artifact(api: HfApi, repo_id: str, file_path: str, artifact_type: str, subset: Optional[str] = None) -> None:
    """
    Validate the artifact of type <artifact_type>, stored in file <file_path>,
    and push an artifact to the specified Hugging Face repository <repo_id> as a dataset subset <subset>.
    api: HfApi instance to use for uploading the file.
    repo_id: str, the Hugging Face repository ID to upload to (e.g., "username/repo_name").
    file_path: str, the local path to the artifact file to be uploaded.
    artifact_type: str, the type of the artifact (e.g., "answers", "queries", etc.) which determines the validation schema to use.
    subset: Optional[str], an optional subset name to organize the artifact within the repository (e.g., "Qwen-7B").
    """
    df = validate_and_load_artifact(file_path, artifact_type)
    
    path_or_fileobj = df_to_bytesio(df) if df is not None else file_path

    fname = os.path.basename(file_path)
    if subset:
        fname = os.path.join(subset, fname)

    api.upload_file(
        path_or_fileobj=path_or_fileobj,
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    print(f"Pushed {artifact_type} '{fname}' to {repo_id}.")


def download_artifact(api: HfApi, repo_id: str, artifact_type: str, subset: Optional[str] = None, filename: Optional[str] = None, save_fpath: Optional[str] = None) -> pd.DataFrame:
    """
    Download and validate the artifact of type <artifact_type> from the Hugging Face repository <repo_id> and eventually store it locally into <filename>.
    api: HfApi instance to use for downloading the file.
    repo_id: str, the Hugging Face repository ID to download from (e.g., "username/repo_name").
    artifact_type: str, the type of the artifact (e.g., "answers", "queries", etc.) which determines the validation schema to use.
    subset: Optional[str], an optional subset name to organize the artifact within the repository (e.g., "Qwen-7B").
    filename: Optional[str], an optional specific filename to download.
    """
    artifact_config: ArtifactSchema = ARTIFACT_REGISTRY.get(artifact_type)
    if not artifact_config:
        raise ValueError(f"Unknown artifact type '{artifact_type}'")

    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    if filename:
        file_to_download = filename
        if subset:
            file_to_download = f"{subset}/{filename}"
        if file_to_download not in files:
            raise ValueError(f"File '{file_to_download}' not found in repository '{repo_id}'")
    else:
        target_files = [f for f in files if any(f.endswith(ext) for ext in artifact_config.extensions)]

        if subset:
            target_files = [f for f in target_files if f.startswith(subset+"/")]

        if not target_files:
            raise ValueError(f"No files found for artifact type '{artifact_type}' in repository '{repo_id}'")

        file_to_download = target_files[0] # first matching file
    local_path = hf_hub_download(repo_id=repo_id, filename=file_to_download, repo_type="dataset")

    df = validate_and_load_artifact(local_path, artifact_type)

    if save_fpath:
        save_path = Path(save_fpath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, 'rb') as src, open(save_path, 'wb') as dst:
            dst.write(src.read())
    os.remove(local_path) # clean up the temporary file
        
    return df

