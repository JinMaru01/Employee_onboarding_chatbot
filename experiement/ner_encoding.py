import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from extend_path import sys
from _lib.database.redis_conn import RedisConn
from _lib.preprocess.data_loader import DataLoader

__conn = RedisConn()

# Load data
file_path = './artifact/data/json/annotated_data_final.json'
load = DataLoader(file_path)
data = load.data

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

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

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

# Create dataset
ner_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)

# Split dataset
train_size = int(0.8 * len(ner_dataset))
test_size = len(ner_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(ner_dataset, [train_size, test_size])

# Save datasets to Redis (optional)
__conn.label_ecoder_save(train_dataset, "ner_train_dataset")
__conn.label_ecoder_save(test_dataset, "ner_test_dataset")

# Save datasets to disk
torch.save(train_dataset, './artifact/data/ner_train_dataset.pt')
torch.save(test_dataset, './artifact/data/ner_test_dataset.pt')

print(f"NER Train dataset size: {len(train_dataset)}")
print(f"NER Test dataset size: {len(test_dataset)}")