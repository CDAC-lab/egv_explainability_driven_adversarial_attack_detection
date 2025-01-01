from datasets import load_dataset
import torch
import wandb
import os
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np

import evaluate
metric = evaluate.load("accuracy")

import gc
gc.collect()

# sst2 = load_dataset

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["sentence"],  padding="max_length", truncation=True)

train_dataset = load_dataset('glue', 'sst2', split='train')
val_dataset = load_dataset('glue', 'sst2', split='validation')
test_dataset = load_dataset('glue', 'sst2', split='test')


train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
val_dataset = val_dataset.remove_columns(['label'])
test_dataset = test_dataset.remove_columns(['label'])
train_dataset = train_dataset.remove_columns(['label'])

MAX_LENGTH = 128
train_dataset = train_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
val_dataset = val_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_API_KEY"] = '98adaaddb053f581b4679c86789fb371b9bb903d'
wandb.init(project='bert-base-uncased-sst2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)




# sst2_encoded = sst2.map(tokenize, batched=True)


num_labels = 2
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))

# print(sst2_encoded["train"].features)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()
print(results)
model.save_pretrained('./sst2_bert_model')
tokenizer.save_pretrained('./sst2_bert_model')
preds_output = trainer.predict(test_dataset)
print(preds_output.metrics)