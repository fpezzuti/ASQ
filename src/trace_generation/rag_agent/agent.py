from dataclasses import dataclass, field
from typing import Callable, Dict

import pyterrier as pt


def _default_model_registry() -> Dict[str, Dict[str, str]]:
    from trace_generation.rag_agent.autorefine import SUPPORTED_MODELS as AUTOREFINE_MODELS
    from trace_generation.rag_agent.gpt_oss import SUPPORTED_MODELS as GPT_OSS_MODELS
    from trace_generation.rag_agent.search_r1 import SUPPORTED_MODELS as SEARCH_R1_MODELS

    return {
        "search_r1": SEARCH_R1_MODELS,
        "autorefine": AUTOREFINE_MODELS,
        "gpt_oss": GPT_OSS_MODELS,
        # add here new supported generators
    }


def _default_builders() -> Dict[str, Callable]:
    from trace_generation.rag_agent.autorefine.builder import build_autorefine
    from trace_generation.rag_agent.gpt_oss.builder import build_gpt_oss
    from trace_generation.rag_agent.search_r1.builder import build_search_r1

    return {
        "search_r1": build_search_r1,
        "autorefine": build_autorefine,
        "gpt_oss": build_gpt_oss,
        # add here the builder of new supported agents
    }

MODEL_REGISTRY = _default_model_registry()
BUILDER_REGISTRY = _default_builders()
SUPPORTED_MODELS = MODEL_REGISTRY
SUPPORTED_MODEL_TYPES = list(MODEL_REGISTRY.keys())

def help_supported_generators(model_type: str) -> None:
    """
    Print the supported generators for an agent family.

    Raises:
        ValueError: If the model type is not registered.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported: {list(MODEL_REGISTRY.keys())}"
        )
    print(f"Agent {model_type} supports the following generators:")
    for gen_key, gen_checkpoint in MODEL_REGISTRY[model_type].items():
        print(f"\t{gen_key}: {gen_checkpoint}")
        
def help_supported_agents() -> None:
    """
    Print the supported agent family keys.
    """
    print("Supported agent families:")
    for model_type in MODEL_REGISTRY.keys():
        print(f"\t{model_type}")
    

def register_agent(model_type: str, supported_models: Dict[str, str], builder: Callable, *, allow_override: bool = False) -> None:
    """
    Register a new agent family and its builder.

    Args:
        model_type: Key for the agent family (e.g., "search_r1").
        supported_models: Mapping of model ids to concrete identifiers.
        builder: Callable that builds an agent instance.
        allow_override: If True, overwrite existing registrations.
    """
    if not allow_override and (model_type in MODEL_REGISTRY or model_type in BUILDER_REGISTRY):
        raise ValueError(f"Model type '{model_type}' is already registered.")
    MODEL_REGISTRY[model_type] = supported_models
    BUILDER_REGISTRY[model_type] = builder
    global SUPPORTED_MODELS, SUPPORTED_MODEL_TYPES
    SUPPORTED_MODELS = MODEL_REGISTRY
    SUPPORTED_MODEL_TYPES = list(MODEL_REGISTRY.keys())

@dataclass(frozen=True)
class RagAgentBuilder:
    """
    Factory for building a RAG agent instance with a validated model type/id.

    Args:
        model_type: Agent family key (e.g., "search_r1", "autorefine").
        model_id: Model identifier within the selected family.
        retrieval_pipeline: PyTerrier transformer used for retrieval.
        agent_k: Top-$k$ documents passed to the agent from retrieval.
    """
    model_type: str
    model_id: str
    retrieval_pipeline: pt.Transformer
    agent_k: int
    retriever_batch_size: int = 1
    retriever_batch_wait_ms: int = 25
    _agent: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        builder = BUILDER_REGISTRY.get(self.model_type)
        if builder is None:
            raise ValueError(f"Unsupported model_type '{self.model_type}'. Supported: {list(BUILDER_REGISTRY.keys())}")
        model = self.resolve_model()
        builder_kwargs = {
            "retrieval_pipeline": self.retrieval_pipeline,
            "agent_k": self.agent_k,
            "model_id": model,
        }
        if self.model_type == "gpt_oss":
            builder_kwargs.update(
                retriever_batch_size=self.retriever_batch_size,
                retriever_batch_wait_ms=self.retriever_batch_wait_ms,
            )
        instance = builder(**builder_kwargs)
        object.__setattr__(self, "_agent", instance)

    def _get_registry(self) -> Dict[str, str]:
        if self.model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model_type '{self.model_type}'. Supported: {list(MODEL_REGISTRY.keys())}")
        return MODEL_REGISTRY[self.model_type]

    def resolve_model(self) -> str:
        registry = self._get_registry()
        if self.model_id not in registry:
            raise ValueError(f"Unsupported model_id '{self.model_id}' for {self.model_type}. Supported: {list(registry.keys())}")
        return registry[self.model_id]

    @property
    def agent(self):
        return self._agent

def build_agent(
    model_type: str,
    model_id: str,
    retrieval_pipeline,
    agent_k: int,
    retriever_batch_size: int = 1,
    retriever_batch_wait_ms: int = 25,
):
    """
    Returns an agent of type <model_type> with generator <model_id>, using the
    top <agent_k> results retrieved using the <retrieval_pipeline> retrieval pipeline.
    """
    agent = RagAgentBuilder(
        model_type=model_type,
        model_id=model_id,
        retrieval_pipeline=retrieval_pipeline,
        agent_k=agent_k,
        retriever_batch_size=retriever_batch_size,
        retriever_batch_wait_ms=retriever_batch_wait_ms,
    )
    return agent.agent


if __name__ == "__main__":
    print("Supported agent families:")
    help_supported_agents()
    print("\nSupported generators by family:")
    for mt in MODEL_REGISTRY.keys():
        help_supported_generators(mt)
