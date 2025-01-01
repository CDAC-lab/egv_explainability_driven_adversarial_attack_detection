from datasets import load_dataset
import torch
import wandb
import os
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification

import gc
gc.collect()

imdb = load_dataset("imdb")
# os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_API_KEY"] = '98adaaddb053f581b4679c86789fb371b9bb903d'
wandb.init(project='bert-base-uncased-imdb')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

imdb_encoded = imdb.map(tokenize, batched=True, batch_size=None)


num_labels = 2
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))

print(imdb_encoded["train"].features)

imdb_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


batch_size = 16
logging_steps = len(imdb_encoded["train"]) // batch_size
training_args = TrainingArguments(output_dir=r"results",
                                  num_train_epochs=5,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1",
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps, save_strategy="epoch")


trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=imdb_encoded["train"],
                  eval_dataset=imdb_encoded["test"])
trainer.train()

results = trainer.evaluate()
print(results)
model.save_pretrained('./imdb_bert_model')
tokenizer.save_pretrained('./imdb_bert_model')
preds_output = trainer.predict(imdb_encoded["test"])
print(preds_output.metrics)