# !pip install transformers peft gdown bitsandbytes rouge-score sacrebleu
# !pip install trl -U -q

import os
import sys

if not os.path.exists('/kaggle/working/efficient-kan'):
#     !git clone https://github.com/Blealtan/efficient-kan

if '/kaggle/working/efficient-kan/src' not in sys.path:
    sys.path.append('/kaggle/working/efficient-kan/src')
    
from efficient_kan import KANLinear

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel, PeftConfig
from datasets import load_dataset, Dataset, load_metric
import json as js
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import bitsandbytes
from datasets import Dataset

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from huggingface_hub import login
login(token=os.getenv("HF_TOKEN"))

# !gdown https://drive.google.com/uc?id=1JNfK2pul14ujIKYfpNECKfi2KkAjg8ZP

with open('/kaggle/working/retriever_dataset.json', 'r') as f:
    dataset = js.load(f)

print(js.dumps(dataset[0], indent=4, ensure_ascii=False))

print(len(dataset), 'samples')

# Load Vietnamese Llama2-7B model source: https://huggingface.co/VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain
model_name = "VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

vocab_size = tokenizer.vocab_size
print(f"Vocabulary size: {vocab_size}")

print(tokenizer.encode('. , + - \n \t'))
encoded_text = tokenizer.encode(dataset[0]['question'])
decoded_text = tokenizer.decode(encoded_text)
print(encoded_text)
print(decoded_text)

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

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# r = 16

# # Set up LoRA configuration
# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     inference_mode=False,
#     r=r,  # Rank parameter for LoRA,
#     target_modules=["q_proj", "v_proj", "k_proj", "gate_proj", "up_proj"],
#     lora_alpha=2*r,  # Alpha parameter for LoRA
#     lora_dropout=0.4,
#     bias="none"
# )

# # Apply LoRA to the model
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config)

# !gdown --folder https://drive.google.com/drive/folders/1Rh5pCYDXGVbedh6z-EnBk9HqKL9EMjRP
    
# # model.enable_input_require_grads()
# lora_config = PeftConfig.from_pretrained('/kaggle/working/weights')

# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, lora_config, '/kaggle/working/weights')

# # Enable training for LoRA layers
# for name, param in model.named_parameters():
#     if 'lora' in name:
#         param.requires_grad = True

lora_config = PeftConfig.from_pretrained('/kaggle/working/llama2_vietnamese_law_model')

adapter_name = '/kaggle/working/llama2_vietnamese_law_model'
model = prepare_model_for_kbit_training(model)
model = PeftModel.from_pretrained(model, adapter_name)

# Enable training for LoRA layers
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True

print(model)

print_trainable_parameters(model)

lengths = []
for data in dataset:
    combined_text = "CÂU HỎI: " + data["question"] + '\n' + "TRẢ LỜI:\n" \
                    + '\n'.join([doc["name"] + '\n' + doc["law"] for doc in data["documents"]]) \
                    + data["answer"]
    lengths.append(tokenizer(combined_text, padding=False, return_tensors="pt").input_ids.shape[1])
    
plt.figure(figsize=(10, 6))
sns.histplot(lengths, bins=20, kde=True)  # kde=True adds a kernel density estimate line
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()

def combine_function(sample):
    combined_text = "CÂU HỎI: " + sample["question"] + '\n' + "TRẢ LỜI:\n" \
                    + '\n'.join([doc["name"] + '\n' + doc["law"] for doc in sample["documents"]]) \
                    + sample["answer"]
    return combined_text

combined_dataset = []
for data in tqdm(dataset):
    combined_dataset.append(combine_function(data))
    
random.seed(42)
random.shuffle(combined_dataset)

train_size = int(0.6 * len(combined_dataset))  # 60% of the data for training
test_size = len(combined_dataset) - train_size

train_dataset, test_dataset = combined_dataset[:train_size], combined_dataset[train_size:]

print(len(train_dataset), len(test_dataset))

tokenized_train_dataset = tokenizer(train_dataset, padding="max_length", truncation=True, return_tensors="pt", max_length=4000)
tokenized_test_dataset = tokenizer(test_dataset, padding="max_length", truncation=True, return_tensors="pt", max_length=4000)

my_train_dataset = Dataset.from_dict({
    'input_ids': tokenized_train_dataset['input_ids'].tolist(),
    'attention_mask': tokenized_train_dataset['attention_mask'].tolist(),
    'labels': tokenized_train_dataset['input_ids'].tolist()  # For causal LM, labels are same as input_ids
})

my_test_dataset = Dataset.from_dict({
    'input_ids': tokenized_test_dataset['input_ids'].tolist(),
    'attention_mask': tokenized_test_dataset['attention_mask'].tolist(),
    'labels': tokenized_test_dataset['input_ids'].tolist()  # For causal LM, labels are same as input_ids
})

# rouge = load_metric("rouge", trust_remote_code=True)
bleu = load_metric("bleu", trust_remote_code=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # BLEU Metrics
    decoded_preds = [pred.split() for pred in decoded_preds]
    decoded_labels = [[label.split()] for label in decoded_labels]
    bleu_output = bleu.compute(predictions=decoded_preds, references=decoded_labels)

    # Perplexity calculation
    perplexity_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = torch.exp(perplexity_loss).item()

    # Combine all metrics into a single dictionary
    return {
        "bleu": bleu_output["bleu"],
        "perplexity": perplexity
    }

def generate_prompt(sample, return_response=True) -> str:
    combined_text = "CÂU HỎI: " + sample["question"] + '\n'

    if return_response:
        full_prompt = combined_text + "TRẢ LỜI:\n" \
                    + '\n'.join([doc["name"] + '\n' + doc["law"] for doc in sample["documents"]]) \
                    + sample["answer"]
        
    return [full_prompt]

# Fine-tuning with qLoRA
model = model.to(device)
model.train()

# Trainer setup
training_args = TrainingArguments(
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-3,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    remove_unused_columns=False,
    logging_dir="./logs",
    output_dir="./llama2_vietnamese_law_model",
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="steps",
    optim="paged_adamw_32bit",
    report_to="none",
    save_steps=20
)

max_seq_length = 4000

trainer = SFTTrainer(
    model=model,
    train_dataset=my_train_dataset,
    peft_config=lora_config,
    max_seq_length=max_seq_length,
    args=training_args,
    dataset_text_field='input_ids'
)

# Fine-tune the model
trainer.train(True)
trainer.save_model()

