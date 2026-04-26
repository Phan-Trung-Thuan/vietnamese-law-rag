# !pip install gdown transformers underthesea rank_bm25 sentence_transformers

# !gdown https://drive.google.com/uc?id=1JNfK2pul14ujIKYfpNECKfi2KkAjg8ZP

import json as js
from transformers import AutoTokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from underthesea import word_tokenize
import re
from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer
import torch

with open('/kaggle/working/retriever_dataset.json', 'r') as f:
    dataset = js.load(f)

print(js.dumps(dataset[0], indent=4, ensure_ascii=False))
print(len(dataset))

# Load Vietnamese Llama2-7B model source: https://huggingface.co/VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain
model_name = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

print(unique_doc_list[0]['law'])
print()
print(preprocess(unique_doc_list[0]['law']))

shortname_len = []
for i in range(len(unique_doc_list)):
    unique_doc_list[i]['tokenized_shortname'] = tokenizer(unique_doc_list[i]['shortname'], padding=False, return_tensors="pt")
    shortname_len.append(unique_doc_list[i]['tokenized_shortname'].input_ids.shape[1])
    
plt.figure(figsize=(10, 6))
sns.histplot(shortname_len, bins=20, kde=True)  # kde=True adds a kernel density estimate line
plt.title('Distribution of Shortname Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()

vectorizer = TfidfVectorizer(max_df=0.85, min_df=5, sublinear_tf=True)
tfidf_matrix = vectorizer.fit_transform([preprocess(doc['law']) for doc in unique_doc_list])

corpus = [preprocess(doc['law']).split() for doc in unique_doc_list]
bm25 = BM25Okapi(corpus)

# For TFIDF

y_pred = []
y_truth = [dataset[i]['documents'] for i in range(len(dataset))]
for i in tqdm(range(len(dataset))):
    query = dataset[i]['question']
    query_vec = vectorizer.transform([preprocess(query)])
    cs = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_docs_indices = np.argsort(cs)[::-1]
    retrieved_docs = []
    for idx in top_docs_indices[:8]:
        retrieved_docs.append(unique_doc_list[idx])
    y_pred.append(retrieved_docs)
    
def inside(text, list_of_text):
    return any(text['name'] == t['name'] for t in list_of_text)

acc1, acc3, acc5 = 0, 0, 0
for i in range(len(y_truth)):
    if all(inside(gt, y_pred[i][:1]) for gt in y_truth[i]):
        acc1 += 1
    
    if all(inside(gt, y_pred[i][:3]) for gt in y_truth[i]):
        acc3 += 1
    
    if all(inside(gt, y_pred[i][:5]) for gt in y_truth[i]):
        acc5 += 1
        
acc1 /= len(y_truth)
acc3 /= len(y_truth)
acc5 /= len(y_truth)
print(acc1, acc3, acc5)

# For BM25

y_pred = []
y_truth = [dataset[i]['documents'] for i in range(len(dataset))]

for i in tqdm(range(len(dataset))):
    query = preprocess(dataset[i]['question']).split()
    
    retrieved_docs = bm25.get_top_n(query, unique_doc_list, n=8)

    y_pred.append(retrieved_docs)
    
def inside(text, list_of_text):
    return any(text['name'] == t['name'] for t in list_of_text)

acc1, acc3, acc5 = 0, 0, 0
for i in range(len(y_truth)):
    if all(inside(gt, y_pred[i][:1]) for gt in y_truth[i]):
        acc1 += 1
    
    if all(inside(gt, y_pred[i][:3]) for gt in y_truth[i]):
        acc3 += 1
    
    if all(inside(gt, y_pred[i][:5]) for gt in y_truth[i]):
        acc5 += 1
        
acc1 /= len(y_truth)
acc3 /= len(y_truth)
acc5 /= len(y_truth)
print(acc1, acc3, acc5)

query = 'Sinh viên có sổ hộ nghèo được hỗ trợ như thế nào về học phí?'
query_vec = vectorizer.transform([preprocess(query)])
    
cs = cosine_similarity(query_vec, tfidf_matrix).flatten()
top_docs_indices = np.argsort(cs)[::-1]

retrieved_docs = []
for idx in top_docs_indices[:5]:
    retrieved_docs.append(unique_doc_list[idx])

print(*['TRẢ LỜI: ' + doc['name'] + '\n' + doc['law'] for doc in retrieved_docs], sep='\n\n')

