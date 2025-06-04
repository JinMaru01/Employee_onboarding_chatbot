from extend_path import sys
from _lib.database.redis_conn import RedisConn
from _lib.preprocess.data_loader import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

import torch
import numpy as np


__conn = RedisConn()

# Load data
file_path = './artifact/data/json/annotated_data.json'
load = DataLoader(file_path)
data = load.data
messages = data['text']

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['intent'])

# Save label encoder to Redis
__conn.label_ecoder_save(label_encoder, "label-encoder")
__conn.label_ecoder_save(labels, "class-labels")

# Load label encoder from Redis
label_encoder = __conn.label_encoder_load("label-encoder")
labels_encodered = __conn.label_encoder_load("class-labels")

# Get label-to-id and id-to-label maps
label2id = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
id2label = {v: k for k, v in label2id.items()}

# Save label2id and id2label to Redis
__conn.label_ecoder_save(label2id, "label2id")
__conn.label_ecoder_save(id2label, "id2label")

# Load label2id and id2label from Redis
label2id = __conn.label_encoder_load("label2id")
id2label = __conn.label_encoder_load("id2label")

# Convert labels to tensor
labels_tensor = torch.tensor(labels_encodered)

# Save labels tensor to Redis
__conn.label_ecoder_save(labels_tensor, "labels_tensor")

# Load labels tensor from Redis
labels_tensor = __conn.label_encoder_load("labels_tensor")

print(f"labels_tensor: {labels_tensor, labels_tensor.shape}")

# Initialize tokenizer and prepare data
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Save tokenizer
__conn.label_ecoder_save(tokenizer, "tokenizer")

# Tokenize with smaller max_length
max_length = 64
encodings = tokenizer(
    messages.tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors='pt'
)

# Save encodings to Redis
__conn.label_ecoder_save(encodings, "classification_encodings_v3")

# Load encodings from Redis
encodings = __conn.label_encoder_load("classification_encodings_v3")
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

samples = data.to_dict(orient='records')  # list of dicts: {tokens, tags}

# Extract all unique NER tags
all_tags = [tag for sample in samples for tag in sample['tags']]
ner_label_encoder = LabelEncoder()
ner_label_encoder.fit(all_tags)

# Save NER label encoder to Redis
__conn.label_ecoder_save(ner_label_encoder.classes_.tolist(), "ner_tags")

# Create label2id and id2label for NER
label2id = {label: idx for idx, label in enumerate(ner_label_encoder.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

# Save label maps to Redis
__conn.label_ecoder_save(label2id, "ner_label2id")
__conn.label_ecoder_save(id2label, "ner_id2label")

# Tokenize and align labels
def tokenize_and_align_labels(tokens, tags, tokenizer, label2id, max_length=64):
    tokenized = tokenizer(
        tokens,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    word_ids = tokenized.word_ids(batch_index=0)
    aligned_labels = []

    prev_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)  # ignore padding
        elif word_idx != prev_word_idx:
            aligned_labels.append(label2id[tags[word_idx]])
        else:
            aligned_labels.append(label2id[tags[word_idx]])  # Use I- or same tag

        prev_word_idx = word_idx

    tokenized["labels"] = torch.tensor([aligned_labels])
    return tokenized

# Prepare tensors
input_ids_list = []
attention_mask_list = []
labels_list = []

for sample in samples:
    tokens = sample["tokens"]
    tags = sample["tags"]

    encoded = tokenize_and_align_labels(tokens, tags, tokenizer, label2id)

    input_ids_list.append(encoded["input_ids"].squeeze())
    attention_mask_list.append(encoded["attention_mask"].squeeze())
    labels_list.append(encoded["labels"].squeeze())

# Stack tensors
input_ids = torch.stack(input_ids_list)
attention_mask = torch.stack(attention_mask_list)
labels = torch.stack(labels_list)