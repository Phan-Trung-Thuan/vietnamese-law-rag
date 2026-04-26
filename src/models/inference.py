# !pip install transformers peft gdown bitsandbytes nltk
# !pip install trl -U -q

import os
import sys

if not os.path.exists('/kaggle/working/efficient-kan'):
#     !git clone https://github.com/Blealtan/efficient-kan

if '/kaggle/working/efficient-kan/src' not in sys.path:
    sys.path.append('/kaggle/working/efficient-kan/src')
    
from efficient_kan import KANLinear

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
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

# Download dataset
# !gdown https://drive.google.com/uc?id=1JNfK2pul14ujIKYfpNECKfi2KkAjg8ZP

with open('/kaggle/working/retriever_dataset.json', 'r') as f:
    dataset = js.load(f)

print(js.dumps(dataset[1], indent=4, ensure_ascii=False))

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

print(model)

def combine_function(sample):
    combined_text = "CÂU HỎI: " + sample["question"] + '\n' + "TRẢ LỜI:\n" \
                    + '\n'.join([doc["name"] + '\n' + doc["law"] for doc in sample["documents"]])
    return combined_text

combined_dataset = []
for data in dataset:
    combined_dataset.append((combine_function(data), data['answer']))
    
random.seed(42)
random.shuffle(combined_dataset)

train_size = int(0.6 * len(combined_dataset))  # 60% of the data for training
test_size = len(combined_dataset) - train_size

train_dataset, test_dataset = combined_dataset[:train_size], combined_dataset[train_size:]

print(len(train_dataset), len(test_dataset))

prompts = []
answers = []
for sample in test_dataset:
    input_text = sample[0]
    output_text = sample[1]
    
    if input_text.endswith('Trân trọng!'):
        input_text = input_text[:-len('Trân trọng!')]
        
    input_text += '\n\nNhư vậy, '
    
    prompts.append(input_text)
    answers.append(output_text)
    
print(prompts[0])

class myDataset(Dataset):
    def __init__(self, prompts, answers):
        self.X = prompts
        self.Y = answers
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
batch_size = 4
my_test_dataset = myDataset(prompts, answers)
my_test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# print(bleu_score_batches)

bleu_score_batches = []

# Continue inference
# score of batch 0..215
bleu_score_batches = [0.5956392169018941, 0.7167149811745789, 0.6531562372839912, 0.6917566293791786, 0.662505705139829, 0.6372638703736072, 0.5976364267140655, 0.7138067813081219, 0.6739393801635332, 0.6324740769499831, 0.7853631530153932, 0.6670975051732803, 0.4605924784771297, 0.6221308605874781, 0.566021807801473, 0.5372988361284821, 0.6846916688285037, 0.7249226662041023, 0.6656242932483253, 0.6855191093809724, 0.6385075420704998, 0.6892237546217106, 0.680138063914408, 0.557041486407766, 0.528008616512863, 0.6332405052849127, 0.666929882686224, 0.6584696045925048, 0.7100553469643723, 0.611009621074835, 0.6701767645308481, 0.6630105980689031, 0.5868861690629396, 0.6142071059983766, 0.6287891877638833, 0.6652689446945645, 0.5658514099350929, 0.7221845815579414, 0.5526428740643228, 0.7555982875996368, 0.7487930574448597, 0.5864917083865929, 0.652429472657261, 0.719457924993392, 0.4350202333699619, 0.5917325250674773, 0.6183022201242034, 0.6250522957123061, 0.6219834566407695, 0.6582436482827108, 0.7335966648027137, 0.5888975872423394, 0.6716199271429567, 0.6270624074438882, 0.6821940007481027, 0.5172504491057968, 0.6882477794815276, 0.6821831464347679, 0.6322594178677097, 0.6768442376497692, 0.6325966541472353, 0.5021659747787461, 0.6819303075185104, 0.46708713781243283, 0.6800031759033518, 0.627657786565951, 0.7378605869368344, 0.7151939595441781, 0.6762374745071225, 0.5862798585305232, 0.5449910241393633, 0.6859380590738118, 0.6826958106565988, 0.7044260448967987, 0.652441246482282, 0.5357880790943519, 0.6919209319020727, 0.520452179870804, 0.5811495061126376, 0.6791376689220832, 0.6760172672080003, 0.5933922439919407, 0.8043455241900094, 0.6268402017493229, 0.6919264550833811, 0.6127679886601551, 0.6760172672080003, 0.5933922439919407, 0.8043455241900094, 0.6268402017493229, 0.6919264550833811, 0.6127679886601551, 0.5747375527797481, 0.6222240700725865, 0.6537247216489225, 0.614720632283863, 0.7238333167524675, 0.5479576805000469, 0.7107497917665133, 0.597677077451731, 0.6486010145975148, 0.7058876476841429, 0.5837953534614291, 0.581782103794138, 0.6582812731041525, 0.6245346274763083, 0.5385400374748992, 0.5937268432806446, 0.5880488301901857, 0.7328095569804952, 0.7051618664081714, 0.6953784998416039, 0.6321718287291133, 0.6673159657588092, 0.6952326641161217, 0.6919892871851324, 0.7366614935561083, 0.6429175584798915, 0.6273885562444816, 0.6517514409588259, 0.6191134364003248, 0.4519176565710454, 0.6506105844943354, 0.5999260931155316, 0.4598382855792399, 0.6650804073887852, 0.7560985347820095, 0.6714951751784888, 0.5861560614602461, 0.579188968654311, 0.7077472265174614, 0.5901180750618701, 0.5525998742968018, 0.7126788997672078, 0.5727272009687538, 0.5626766936461576, 0.6204405101325996, 0.7129271835059163, 0.5879071450294561, 0.5611984904015577, 0.5474509533081884, 0.6118429230318931, 0.4422937868381822, 0.6711425946578288, 0.7371729473044242, 0.42306867480557186, 0.7238489817866232, 0.6030122179712373, 0.7116293787839783, 0.6818746535377352, 0.713298013216964, 0.5371144042599167, 0.7442048228694986, 0.49893260572641296, 0.6451195293246818, 0.6479458414629694, 0.6785104488732591, 0.6090737146070214, 0.6535502975241387, 0.5956424931138404, 0.5598515191342839, 0.6243468787856716, 0.6467127617447361, 0.7299412038182256, 0.5174261882775502, 0.6945175599512815, 0.6850819500882545, 0.7418763597617963, 0.6246479288614787, 0.7265213945244661, 0.3960332962508236, 0.6280920911896596, 0.5480143827697719, 0.7200585292606679, 0.6667909042790255, 0.6541633241470333, 0.5595122826853726, 0.5379688342716976, 0.6305982085774169, 0.6260635646206163, 0.6583957608282591, 0.5117243510514886, 0.68049812491938, 0.4987873804823763, 0.6063189598873291, 0.5362609443670528, 0.6595242347725805, 0.6381027143648546, 0.5789688543204106, 0.672476625560296, 0.5932881918736989, 0.5661822555304092, 0.6037954495445758, 0.7124506887736424, 0.5178554568178826, 0.5769140747739552, 0.5480709686314879, 0.6008539068483433, 0.7225682335062724, 0.7314505867040595, 0.7418593669382674, 0.5121416598353448, 0.6740282092269823, 0.6127148243678646, 0.5731469361504491, 0.5970923882169421, 0.6517973333083531, 0.6409413989158825, 0.748241508530749, 0.7171788750677519, 0.5053660364313781, 0.5819848445234316]

model.eval()

for i, (X_batch, Y_batch) in enumerate(tqdm(my_test_dataloader)):
    # Continue inference
    if i <= 215:
        continue
    
    tokenized_text = tokenizer(X_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    pred = model.generate(**tokenized_text, eos_token_id=tokenizer.eos_token_id, max_new_tokens=500).cpu()
    generated = tokenizer.batch_decode(pred, skip_special_tokens=True)
    
    tokenized_references = [[tokenizer(X_batch[i] + '\n' + Y_batch[i])['input_ids']] for i in range(len(Y_batch))]
    tokenized_generated = [tokenizer(gen)['input_ids'] for gen in generated]

    bleu_scores_samples = []
    for ref, hyp in zip(tokenized_references, tokenized_generated):
        score = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
        bleu_scores_samples.append(score)
        
        del score
        del ref
        del hyp

    average_bleu_score = sum(bleu_scores_samples) / len(bleu_scores_samples)
#     print(average_bleu_score)
    bleu_score_batches.append(average_bleu_score)
    
    del X_batch
    del Y_batch
    del tokenized_text
    del pred
    del generated
    del tokenized_references
    del tokenized_generated
    del bleu_scores_samples
    torch.cuda.empty_cache()
    
print(sum(bleu_score_batches) / len(bleu_score_batches))

sample = dataset[122]
prompt = "CÂU HỎI: " + sample["question"] + '\n' + "TRẢ LỜI:\n" \
        + '\n'.join([doc["name"] + '\n' + doc["law"] for doc in sample["documents"]])

if prompt.endswith('Trân trọng!'):
    prompt = prompt[:-len('Trân trọng!')]
    
prompt += '\n\nNhư vậy, '
    
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=800)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print(prompt)

