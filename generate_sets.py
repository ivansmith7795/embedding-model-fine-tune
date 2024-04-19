import json

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode

from llama_cpp import Llama

TRAIN_FILES = 'training'
VAL_FILES = 'validation'

TRAIN_CORPUS_FPATH = "./data/train_corpus.json"
VAL_CORPUS_FPATH = "./data/val_corpus.json"

def build_llm():

    llm = Llama(
        # set the path to a pre-downloaded model instead of model_url
        model_path='../Llama2/llama.cpp/ggml-model-7b-chat-hf-f16.bin',
        # we're using GPU, so we need to set the number of layers to offload to CUDA
        n_gpu_layers=100,
        # our context size is large, so we'll need to increase it
        n_ctx=3900,
        # Increate the logging to verbose
        verbose=True,
    )
    return llm

representation_model = build_llm()

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_dir=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes

train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)


from llama_index.finetuning import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset,
)

QA_PROMPT_TEMPLATE = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are an HR analyst. Your task is to come up with \
{num_questions_per_chunk} questions someone may ask based on the context provided.\
Restrict the questions to the \
context information provided. Include only one question in the response. \
Do not include any other information. Do not include number prefixes\
Do not prefix the question with Question, Question: or Q1 or any other notation. Only include the question text. \
Do not include multiple choice options. Only include the single question in the response. Do not provide an explaination for the question."
"""

train_dataset = generate_qa_embedding_pairs(nodes=train_nodes, llm=representation_model, num_questions_per_chunk=1, qa_generate_prompt_tmpl=QA_PROMPT_TEMPLATE)
train_dataset.save_json("train_dataset.json")

val_dataset = generate_qa_embedding_pairs(nodes=val_nodes, llm=representation_model, num_questions_per_chunk=1, qa_generate_prompt_tmpl=QA_PROMPT_TEMPLATE)
val_dataset.save_json("val_dataset.json")