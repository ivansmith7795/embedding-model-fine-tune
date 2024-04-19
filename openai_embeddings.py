import argparse
import os
from sentence_transformers import SentenceTransformer
from eval_utils import evaluate
import json
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from eval_utils.embedding import *
import pickle
from eval_utils.utils import *
from torch.utils.data import DataLoader
import numpy as np

TRAIN_DATASET_FPATH = 'train_dataset.json'
VAL_DATASET_FPATH = 'val_dataset.json'

openai.api_key = "KEY"

with open(TRAIN_DATASET_FPATH, 'r+') as f:
    train_dataset = json.load(f)

with open(VAL_DATASET_FPATH, 'r+') as f:
    val_dataset = json.load(f)

corpus = val_dataset['corpus']
queries = val_dataset['queries']
relevant_docs = val_dataset['relevant_docs']

encoder=OpenAIEncoder()
print('Embed queries...')

query_embeddings=get_embedding_dict(queries,encoder)
print('Embed corpus...')

corpus_embeddings=get_embedding_dict(corpus,encoder)

# Save the list
with open('data/val_query_openai.pkl', 'wb') as f:
    pickle.dump(query_embeddings, f)

with open('data/val_corpus_openai.pkl', 'wb') as f:
    pickle.dump(corpus_embeddings, f)