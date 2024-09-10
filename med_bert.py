#!/usr/bin/env python
# coding: utf-8

# In[29]:


from tqdm import tqdm
import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, AutoModel, \
    TrainingArguments, Trainer

"""
d = {'text': [], 'filename': []}

for file in tqdm(os.listdir('data/japanese_med/japanese_crawl')):
    try:
        with open(os.path.join('data/japanese_med/japanese_crawl', file), 'r') as f:
            text = f.read()
            d['text'].append(text) 
            d['filename'].append(file)
    except Exception as e:
        print(e)
json.dump(d, open('data/japanese_med/raw.json', 'w'), ensure_ascii=False, indent=2)
"""

ds = Dataset.from_dict(json.load(open('data/japanese_med/raw.json')))
ds = ds.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")


def preprocess_function(examples):
    return tokenizer(examples["text"])


tokenized_ds = ds.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=ds["train"].column_names,
)

block_size = 512


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=4)
model = AutoModelForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})


data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)




training_args = TrainingArguments(
    output_dir="data/japanese_med/ja_med_bert",
    eval_strategy="epoch",
    learning_rate=1e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps=1,
    save_strategy="epoch",
    save_total_limit=3,
    per_device_train_batch_size=64
)

# Number of samples you want to use for testing
debug = False
if debug:
    train_dataset = lm_dataset["train"].select(range(100))  # First 100 samples
    eval_dataset = lm_dataset["test"].select(range(50))  # First 50 samples
else:
    train_dataset = lm_dataset["train"]
    eval_dataset = lm_dataset["test"]

trainer = Trainer(
    model=model,
    args=training_args,
    #train_dataset=small_train_dataset,
    #eval_dataset=small_eval_dataset,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
