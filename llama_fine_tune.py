from llama_index.finetuning import EmbeddingAdapterFinetuneEngine
from llama_index.embeddings import resolve_embed_model
from llama_index.finetuning import EmbeddingQAFinetuneDataset

import torch

base_embed_model = resolve_embed_model("local:BAAI/bge-base-en-v1.5")
train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")

finetune_engine = EmbeddingAdapterFinetuneEngine(
    train_dataset,
    base_embed_model,
    model_output_path="model",
    # bias=True,
    epochs=20,
    verbose=True,
    # optimizer_class=torch.optim.SGD,
    # optimizer_params={"lr": 0.01}
)

finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()