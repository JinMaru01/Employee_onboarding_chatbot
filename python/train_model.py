# Hugging Face NER Training Pipeline for Custom Entities

from datasets import Dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)
from sklearn.model_selection import train_test_split
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import torch
import json

# Step 1: Define entity labels
entity_types = [
    "Topic", "Duration", "Date", "Time", "Budget", "Participant_count",
    "Location", "Person", "Organization", "Skill_level", "Position",
    "Leave_type", "Contact_info"
]
label_list = ["O"] + [f"B-{e}" for e in entity_types] + [f"I-{e}" for e in entity_types]
label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {i: l for l, i in label_to_id.items()}

# Step 2: Load and process the data
filepath = "./data/all_intents_ner.json"
with open(filepath) as f:
    raw_data = json.load(f)

examples = []
for tokens, labels in raw_data:
    if len(tokens) == len(labels):
        examples.append({"tokens": tokens, "ner_tags": labels})

dataset = Dataset.from_list(examples)
train_test = dataset.train_test_split(test_size=0.2)
datasets = DatasetDict({
    "train": train_test["train"],
    "validation": train_test["test"]
})

# Step 3: Tokenization and alignment
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(example["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label_to_id.get(example["ner_tags"][word_idx], 0))
        else:
            label = example["ner_tags"][word_idx]
            if label.startswith("B-"):
                label = label.replace("B-", "I-")
            labels.append(label_to_id.get(label, 0))
        previous_word_idx = word_idx
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=False)

# Step 4: Model and training setup
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(label_list),
    id2label=id_to_label,
    label2id=label_to_id
)

data_collator = DataCollatorForTokenClassification(tokenizer)

# Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id_to_label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id_to_label[p] for (p, l) in zip(pred, label) if l != -100]
                  for pred, label in zip(predictions, labels)]

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds)
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./model/ner_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Step 5: Train
trainer.train()

# Save final model
trainer.save_model("./model/ner_model")
