import traceback
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from nanogcg.gcg_embedding import run_embedding_attack, EmbeddingGCGConfig


MODEL_PATH = "/u/anp407/Workspace/Huggingface/Qwen/Qwen3-Embedding-0.6B"


def main():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

        print("Loading model (this may take a while)...")
        # Use device_map auto if available to avoid OOM on a single device
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

        config = EmbeddingGCGConfig(
            num_steps=3,
            search_width=8,
            topk=32,
            n_replace=1,
            buffer_size=2,
            verbosity="INFO",
        )

        doc = "This is a short document used for a quick smoke test."
        query = "short query for smoke test"

        print("Running embedding attack (quick)...")
        result = run_embedding_attack(model, tokenizer, doc, query, config=config)

        print("Result:")
        print("Best score:", result.best_score)
        print("Best string:", result.best_string)
        print("Scores list:", result.scores)
    except Exception:
        print("Smoke test failed with exception:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
