SUPPORTED_MODELS = {
	"Qwen-3B": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-em-ppo-v0.3",
	"Qwen-7B": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
	"Qwen-14B": "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-ppo-v0.2",
}
from .builder import build_search_r1

__all__ = ["SUPPORTED_MODELS", "build_search_r1"]
