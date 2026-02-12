<div align="center">
  <h1>🤖 ASQ: Agentic Search Queryset </h1>
  <p><strong> A dataset capturing RAG agents' search behaviours.</strong></p>
</div>

<div align="center">
  
[![HF Dataset](https://img.shields.io/badge/Dataset-AgenticSearchQueryset%2FASQ-FF9D00)](https://huggingface.co/datasets/AgenticSearchQueryset/ASQ)
[![Paper](https://img.shields.io/badge/arXiv-2602.XXXXX-B31B1B.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

This repository contains the tools to reconstruct the dataset, access the traces, and perform behavioral analysis.

## 🗂️ Repository Structure
This repository provides a modular framework for generating, storing, and analszing agentic search traces.
Below is the organizational structure of the source code:
```text
ASQ/
├── README.md
├── requirements.txt
├── setup.sh
└── src/ # 👈 main development root
    ├── quickstart_trace_generation.ipynb   # trace generation walkthrough
    ├── quickstart_data_access.ipynb        # artifact loading/analysis walkthrough
    ├── artifacts/                          # artifact schemas and loaders
    ├── data_utils/                         # dataset/retrieval helpers
    └── trace_generation/                   # agent setup, execution, extraction
```

## ⚙️ Setup & Requirements
Ensure you have **Java** installed to support the retrieval components.
Use the following configuration to set up your local environment:

```bash
# create conda environment
conda create -n asq python=3.10 numpy==1.26.4 pandas -y
conda activate asq

# install dependencies
pip install -r requirements.txt
```

## 🚀 Quickstarts

Use these notebooks to get started quickly with trace generation and artifact loading and analysis.

- `src/quickstart_trace_generation.ipynb`: Generate traces.
- `src/quickstart_data_access.ipynb`: Load trace artifacts and run basic analytics.


## 🏗️ Generating your ASQ-like dataset
We now describe how to **construct an ASQ-like dataset**. For a practical walkthrough, see `src/quickstart_trace_generation.ipynb`.

To **reproduce the construction of ASQ**, use the `tutorial_main.py` as shown in the aforementioned walkthrough about trace generation.

### 1. Setting up the agent

To instantiate one of the supported agents use the `build_agent()`method as follows.

```python
from trace_generation.rag_agent.agent import build_agent

agent = build_agent(
    model_type="search_r1", # agent family  
    model_id="Qwen-7B", # generator
    retrieval_pipeline=retrieval_pipeline, # PyTerrier retrieval pipeline
    agent_k=3, # number of retrieved documents
)
```

In the following, we describe how to set the agent family, the generator and the retrieval pipeline

### 1.1 Setting up the generator
At the moment we support the following configurations:
- `model_type`: agent family (`search_r1` or `autorefine`).
- `model_id`: generator checkpoint to use (see supported lists below).

The list of supported `model_type` agent families and of their `model_id` supported models can be seen using the `trace_generation.help_supported_agents()` and `trace_generation.help_supported_generators()` functions. 


### 1.2 Setting up the retrieval pipeline
Use either an existing `pyterrier.Transformer` retrieval pipeline, or create your custom retriever by implement a [PyTerrier transformer](https://pyterrier.readthedocs.io/en/latest/terrier/api.html#pyterrier.terrier.Retriever) that:
1. Accepts input rows with `qid` and `query`.
2. Returns rows with `qid`, `docno`, `rank`, `score`, and `text`.
3. Returns `title` when available.

See `src/quickstart_trace_generation.ipynb` for some practical examples.

### 2. Preparing the source organic queryset
The next step is to prepare the input query set.

- If the input query set is an [IRDS](https://ir-datasets.com/) dataset:
	1. You can use our default `load_dataset()` method passing the dataset's IRDS name without the `irds:` prefix.
- If it is a custom dataset:
	- build a queries DataFrame with `qid` and `query` columns.

See `src/quickstart_trace_generation.ipynb` for some practical examples.

### 3. Collecting Traces (Execution of the Agentic Run)

Use the following command to start generation:
```python
res = agent(queries_df)
```

> **Note**: This process is computationally intensive. Pre-computed traces can be downloaded from Hugging Face Datasets: [ASQ dataset](https://huggingface.co/datasets/AgenticSearchQueryset/ASQ).



### 4. Creating the extraction pipeline
By default traces are returned by the agent as an in-memory DataFrame and are **not automatically saved** to disk.

The raw agent output DataFrame contains `qid`, `query`, `answer`, `all_queries`, `all_docids` (and optionally `all_thoughts`) columns.  
Use the `DataFrameExtractor` class to converts these into TSV artifacts and appends to existing files and persist artifacts in ASQ format.

```python
extraction_order = [ExtractionType.ANSWERS, # define the extraction order
                    ExtractionType.ITER_QUERIES,
                    ExtractionType.RETRIEVED_DOCS,
                    ExtractionType.THOUGHTS]

extractor = DataFrameExtractor(extraction_order=extraction_order) # instantiate the extractor

results_df = agent(queries_df)  # in-memory trace output

extractor.extract_all_and_save(results_df, output_paths) # persist
```


##  🗄️ Accessing the ASQ Dataset

Traces are stored in a **hierarchical directory structure** for easy filtering.

Directory `traces/{dataset}/{retriever_config}/{agent_family}/{model}/` stores collections of traces for the following configuration:
- `dataset`: queryset internal identifier or irds identifier (e.g., "hotpotqa-test")
- `retriever_config`: retrieval pipeline configuration (e.g., "BM25_k1000_electra_k3")
- `agent_family`: one of the supported agent families (e.g., "search_r1")
- `model`: one of the supported generators of `agent_family` (e.g., "Qwen-7B")


To simplify path resolution and loading, use the `artifacts.TraceCollectionConfig` and `OutputPaths` classes, and the `config.build_output_paths()` methods.

See `src/quickstart_data_access.ipynb` for a step-by-step example on how to use them.

### Loading Full Trace Collections

To load full trace collection:
```python
from artifacts import TraceCollection

trace = TraceCollection(config)
trace.load_data(load_docs=True)
```

### Loading Individual Artifacts

Artifacts can be also loaded in memory individually from a filepath.

Instead of manually building paths, you can use `build_output_paths()`:
```python
from trace_generation.output_paths import OutputPaths
output_paths: OutputPaths = config.build_output_paths()
```

Then, you can load each artifact directly using their load function.

```python
from artifacts.answers import load_answers
from artifacts.iter_queries import load_iter_queries
from artifacts.retrieved_docs import load_retrieved_docs

answers = load_answers(output_paths.answers)
iter_queries = load_iter_queries(output_paths.iter_queries)
retrieved_docs = load_retrieved_docs(output_paths.retrieved_docs)
```

You can also pass any custom file path instead of `output_paths.*`.
```python
answers = load_answers(f"{base_dir}/answers.tsv")
iter_queries = load_iter_queries(f"{base_dir}/iter_queries.tsv")
retrieved_docs = load_retrieved_docs(f"{base_dir}/retrieved_docs.tsv")
```

## 📈 Analysing Traces

After loading artifacts or traces, you can use several methods to quickly analyse them.

E.g.,:
- `AnswersArtifact.nunique_qids()`: count answered queries.
- `SyntheticQueriesArtifact.iteration_counts()`: see how many queries required each iteration.
- `SyntheticQueriesArtifact.query_trace_lengths()`: map each `qid` to its max iteration.
- `RetrievedDocsArtifact.docids_for_qid()`: inspect the docids retrieved documents for a `qid`.
- `RetrievedDocsArtifact.format_runs_into_trec_format()`: create a TREC run and access it with `get_it_run()`.

Each artifact class also provides a `.helper()` method that prints the available methods and their purpose.


See `src/quickstart_data_access.ipynb` for more details and examples.


## 🧱 Extending ASQ with Other Agent Families

At the moment we support the following models:

| Agent Family |Supported Generators | Note |
| :--- | :--- | :--- |
| `search_r1` | [Qwen-3B](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3), [Qwen-7B](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo), [Qwen-14B](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-ppo-v0.2)  | [original paper](https://arxiv.org/abs/2503.09516) |
| `autorefine` | [Qwen-3B](https://huggingface.co/yrshi/AutoRefine-Qwen2.5-3B-Instruct) | [original paper](https://arxiv.org/abs/2505.11277) |


To add a **new agent type**, follow these steps:
1. Implement the agent package (with `SUPPORTED_MODELS` and a builder function) as a subpackage of `trace_generation/rag_agent/`.
2. Register the new agent type using `trace_generation.rag_agent.register_agent()`.


## ⚖️ License 
The ASQ dataset is released under the [*MIT License*](https://opensource.org/license/mit). Individual source datasets may have their own licenses.


## 🛠️ Ethics Statement
ASQ is derived from publicly available datasets and is intended solely for research on agentic search behaviour.
The authors do not endorse or assume responsibility for the content or any biases present in the traces.
The contents of these traces should not be interpreted as representing the views of the researchers or their institutions.
Users are advised to apply safety and content filters when using them.


## 🔗 Citation

If you find our work useful, please cite it as follows:

```bibtex
@misc{fpezzuti2026asq,
      title={A Picture of Agentic Search}, 
      author={Pezzuti, Francesca and Frieder, Ophir and Silvestri, Fabrizio and MacAvaney, Sean and Tonellotto, Nicola},
      year={2026},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={}, 
}
```
