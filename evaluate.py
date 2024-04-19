from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.schema import TextNode
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from eval_utils.embedding import SentenceTransformerEncoder
from eval_utils.embedding import OpenAIEncoder
from eval_utils.embedding import FAEEncoder

from tqdm.notebook import tqdm
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

from eval_utils import evaluate

VAL_DATASET_FPATH = 'val_dataset.json'
with open(VAL_DATASET_FPATH, 'r+') as f:
    val_dataset = json.load(f)

#val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")
encoder=OpenAIEncoder()
print('OpenAI/text-embedding-ada-002 hit rate:',evaluate.evaluate(val_dataset,encoder, if_load=True))


encoder=SentenceTransformerEncoder(model_name="BAAI/bge-base-en-v1.5")
print('BAAI/bge-base-en-v1.5 pretrained hit rate:',evaluate.evaluate(val_dataset,encoder))


encoder=FAEEncoder(model_name="BAAI/bge-base-en-v1.5",if_load=True)
print('FAE pre-trained hit rate:',evaluate.evaluate(val_dataset,encoder))


bbai_eval_list=np.load('evaluations/bge-small-en_eval.npy')

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(bbai_eval_list, marker='o', linestyle='-', color='grey', label=' Fine-tuned BAAI/bge-small-en')
openai_eval = 0.566839378238342
ax.axhline(y=openai_eval, color='g', linestyle='--', label='OpenAI/text-embedding-ada-002')
title_font = {'color': 'black', 'size': 15} 
ax.set_xlabel('Training Epoch')
ax.set_ylabel('Hit Rate')
ax.set_title('Hit Rate vs Training Epoch', fontdict=title_font)
legend = ax.legend(loc='lower right') 
legend.get_texts()[0].set_color('grey') 
legend.get_texts()[1].set_color('g')  

ax.set_ylim(0.4, 0.6)

plt.savefig('evaluation.png')

fig.clf()

bbai_eval_list=np.load('evaluations/bge-small-en_eval.npy')
fae_eval_list=np.load('evaluations/fae_eval.npy')

fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(bbai_eval_list, marker='o', linestyle='-', color='grey', label=' Fine-tuned BAAI/bge-small-en')
openai_eval = 0.566839378238342
ax.axhline(y=openai_eval, color='g', linestyle='--', label='OpenAI/text-embedding-ada-002')
ax.plot(fae_eval_list, marker='o', linestyle='-', color='#2D72CF', label='EnsembleModel')
title_font = {'color': 'black', 'size': 15} 
ax.set_xlabel('Training Epoch')
ax.set_ylabel('Hit Rate')
ax.set_title('Hit Rate vs Training Epoch', fontdict=title_font)
legend = ax.legend(loc='lower right') 
legend.get_texts()[0].set_color('grey') 
legend.get_texts()[1].set_color('g')  
legend.get_texts()[2].set_color('#2D72CF')  

ax.set_ylim(0.4, 0.6)

plt.savefig('evaluation_fae.png')