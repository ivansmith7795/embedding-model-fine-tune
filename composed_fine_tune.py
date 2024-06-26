import argparse
import os
from sentence_transformers import SentenceTransformer
from eval_utils import evaluate
import json
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from eval_utils.loss import *
from eval_utils.embedding import *
import pickle
from eval_utils.utils import *
from torch.utils.data import DataLoader
import numpy as np

seed=100
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)

batch_size = 20
device = 'cuda:0'


AUGMENTED_MODEL_NAME="BAAI/bge-base-en-v1.5"


# Datasets
TRAIN_DATASET_FPATH = 'train_dataset.json'
VAL_DATASET_FPATH = 'val_dataset.json'
with open(TRAIN_DATASET_FPATH, 'r+') as f:
    train_dataset = json.load(f)
with open(VAL_DATASET_FPATH, 'r+') as f:
    val_dataset = json.load(f)
corpus = train_dataset['corpus']
queries = train_dataset['queries']
relevant_docs = train_dataset['relevant_docs']




results_folder = f'save/FAE/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


# Load the saved openai embeddings
with open('data/train_query_openai.pkl', 'rb') as f:
    train_query_openai_dict = pickle.load(f)

with open('data/train_corpus_openai.pkl', 'rb') as f:
    train_corpus_openai_dict = pickle.load(f)
    
examples = []
for query_id, query in queries.items():
    node_id = relevant_docs[query_id][0]
    text = corpus[node_id]
    example = FAEExample(texts=[query, text], given_embeddings=[train_query_openai_dict[query_id],train_corpus_openai_dict[node_id]])
    examples.append(example)
loader = DataLoader(examples, batch_size=batch_size)


encoder=OpenAIEncoder()
hit_rate_temp = evaluate.evaluate(val_dataset, encoder,if_load=True)
print(f"OPENAI - Hit Rate: {hit_rate_temp:.6f}")


model = SentenceTransformer(AUGMENTED_MODEL_NAME).to(device)
model.smart_batching_collate= fae_smart_batching_collate(model)
loss=FAEMultipleNegativesRankingLoss(model)


MAX_EPOCHS = 20
previous_hit_rate = 0
hit_rates = []

encoder=FAEEncoder(model_name=AUGMENTED_MODEL_NAME)
hit_rate_temp = evaluate.evaluate(val_dataset, encoder,if_load=True)
print(f"Pretrained model - Hit Rate: {hit_rate_temp:.6f}")
hit_rates.append(hit_rate_temp)
previous_hit_rate=hit_rate_temp


for epoch in range(1, MAX_EPOCHS + 1):
    if epoch != 1:
        model = SentenceTransformer(os.path.join(results_folder, f'epoch_{epoch-1}')).to(device)
        model.smart_batching_collate= fae_smart_batching_collate(model)
        loss=FAEMultipleNegativesRankingLoss(model)


    model_output_path = os.path.join(results_folder, f'epoch_{epoch}')
    print(f"Epoch {epoch} - Training Start")
    model.fit(train_objectives=[(loader, loss)], 
              epochs=1, 
              show_progress_bar=True,
              evaluation_steps=50,
              warmup_steps=int(len(loader) * 0.1), 
              output_path=model_output_path)
    

    print(f"Epoch {epoch} - Evaluation Start")
    encoder=FAEEncoder(model_name=model_output_path)
    hit_rate_temp = evaluate.evaluate(val_dataset, encoder, if_load=True)
    hit_rates.append(hit_rate_temp)

    print(f"Epoch {epoch} - Hit Rate: {hit_rate_temp:.6f}")

    ### Conditions for early stop
    if hit_rate_temp > previous_hit_rate:
        previous_hit_rate = hit_rate_temp
        consecutive_no_improve_epochs = 0
    else:
        consecutive_no_improve_epochs += 1

    if consecutive_no_improve_epochs >= 4:
        print("Early stopping due to no improvement in hit rate.")
        break

print(hit_rates)
with open('evaluations/fae_eval.npy', 'wb') as f:
    np.save(f, hit_rates)

    