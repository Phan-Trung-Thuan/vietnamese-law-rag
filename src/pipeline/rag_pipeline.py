# !pip install transformers peft gdown bitsandbytes nltk underthesea rank_bm25
# !pip install trl -U -q

import os
import sys

if not os.path.exists('/kaggle/working/efficient-kan'):
#     !git clone https://github.com/Blealtan/efficient-kan

if '/kaggle/working/efficient-kan/src' not in sys.path:
    sys.path.append('/kaggle/working/efficient-kan/src')
    
from efficient_kan import KANLinear

import nltk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel, PeftConfig
from datasets import load_dataset, Dataset, load_metric
import json as js
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import bitsandbytes
from datasets import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from underthesea import word_tokenize
import re

# !gdown https://drive.google.com/uc?id=1JNfK2pul14ujIKYfpNECKfi2KkAjg8ZP

with open('/kaggle/working/retriever_dataset.json', 'r') as f:
    dataset = js.load(f)

print(js.dumps(dataset[0], indent=4, ensure_ascii=False))
print(len(dataset))

# !git clone https://github.com/stopwords/vietnamese-stopwords

with open('/kaggle/working/vietnamese-stopwords/vietnamese-stopwords.txt', 'r', encoding='utf-8') as file:
    stop_words = file.read()
    stop_words = stop_words.split('\n')


def preprocess(text):
    """
    Preprocesses a Vietnamese text by:
    1. Lowercasing the text
    2. Removing special characters and numbers
    3. Tokenizing the text into words
    4. Removing Vietnamese stopwords
    
    Args:
        text (str): The input text to preprocess.
        
    Returns:
        str: The preprocessed text as a single string of tokens.
    """
    
    # Step 1: Lowercasing
    text = text.lower()
    
    if text.endswith('Trân trọng.'):
        text = text[:len('Trân trọng.')]
    
    # Step 3: Tokenization
    tokens = word_tokenize(text, format='text')
    
    # Step 4: Stopword removal
    tokens = [word for word in tokens.split() if word not in stop_words]
    
    # Return the processed text as a single string
    return ' '.join(tokens)

raw_doc_list = []
unique_doc_list = []
doc_set = set()
for i in tqdm(range(len(dataset))):
    tmp = dataset[i]['documents']
    
    raw_doc_list.extend(tmp)
    unique_doc_list.extend([item for item in tmp if preprocess(item['law']) not in doc_set])
        
    for item in tmp:
        doc_set.add(preprocess(item['law']))
        
print(len(raw_doc_list))
print(len(unique_doc_list))

# vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, sublinear_tf=True)
# tfidf_matrix = vectorizer.fit_transform([preprocess(doc['law']) for doc in unique_doc_list])

corpus = [preprocess(doc['law']).split() for doc in unique_doc_list]
bm25 = BM25Okapi(corpus)

# Load Vietnamese Llama2-7B model source: https://huggingface.co/VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain
model_name = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    use_cache=False,
    device_map="auto"
)

model.model.layers = model.model.layers[:16]

# Download LoRA weights
# !gdown --folder https://drive.google.com/drive/folders/1Rh5pCYDXGVbedh6z-EnBk9HqKL9EMjRP

lora_config = PeftConfig.from_pretrained('/kaggle/working/llama2_vietnamese_law_model')

adapter_name = '/kaggle/working/llama2_vietnamese_law_model'
model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, adapter_name)

prompt = "Sinh viên là người dân tộc Ơ Đu thì sẽ được miễn học phí?"

# For TFIDF
# query_vec = vectorizer.transform([preprocess(prompt)])
# cs = cosine_similarity(query_vec, tfidf_matrix).flatten()
# top_docs_indices = np.argsort(cs)[::-1]
# retrieved_docs = []
# for idx in top_docs_indices[:3]:
#     retrieved_docs.append(unique_doc_list[idx])
    
# For BM25
query = preprocess(prompt).split()
retrieved_docs = bm25.get_top_n(query, unique_doc_list, n=1)

def combine_function(prompt, docs):
    combined_text = "CÂU HỎI: " + prompt + '\n' + "TRẢ LỜI:\n" \
                    + '\n'.join([doc["name"] + '\n' + doc["law"] for doc in docs])
    return combined_text

combined_prompt = combine_function(prompt, retrieved_docs)

if combined_prompt.endswith('Trân trọng!') or combined_prompt.endswith('Trân trọng.') :
    combined_prompt = combined_prompt[:-len('Trân trọng!')]

combined_prompt += '\n\nNhư vậy, '

print(combined_prompt)

inputs = tokenizer(combined_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=800)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

