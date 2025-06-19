import torch
from extend_path import sys
from transformers import AutoTokenizer
from _lib.preprocess.data_loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from _lib.database.postgres_conn import PostgresConn

pg = PostgresConn()

# Load data
file_path = './artifact/data/json/annotated_data.json'
load = DataLoader(file_path)
data = load.data
messages = data['text']

# Save label encoder and its mappings
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['intent'])

pg.save_artifact("label-encoder", label_encoder, "label_encoder")
pg.save_artifact("class-labels", labels, "label_ids")

# Save label2id and id2label
label2id = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
id2label = {v: k for k, v in label2id.items()}

pg.save_artifact("label2id", label2id, "label_map")
pg.save_artifact("id2label", id2label, "label_map")

# Save labels tensor
labels_tensor = torch.tensor(labels)
pg.save_artifact("labels_tensor", labels_tensor, "tensor")

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
pg.save_artifact("tokenizer", tokenizer, "tokenizer")

# Tokenized encodings
encodings = tokenizer(
    messages.tolist(),
    truncation=True,
    padding=True,
    max_length=64,
    return_tensors='pt'
)
pg.save_artifact("classification_encodings_v3", encodings, "encodings")

samples = data.to_dict(orient='records')

# Save NER label encodings
ner_tags = [tag for sample in samples for tag in sample["tags"]]
ner_encoder = LabelEncoder()
ner_encoder.fit(ner_tags)

pg.save_artifact("ner_tags", ner_encoder.classes_.tolist(), "ner_classes")

ner_label2id = {label: idx for idx, label in enumerate(ner_encoder.classes_)}
ner_id2label = {idx: label for label, idx in ner_label2id.items()}

pg.save_artifact("ner_label2id", ner_label2id, "label_map")
pg.save_artifact("ner_id2label", ner_id2label, "label_map")

pg.close()