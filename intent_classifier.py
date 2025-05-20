import torch

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification
from _lib.database.redis_conn import RedisConn
from _lib.models.Intent_Classification import IntentClassifier

# Initialize Redis connection
__conn = RedisConn()

# Load train and test datasets from Redis
train_dataset = __conn.label_encoder_load("classification_train_dataset")
test_dataset = __conn.label_encoder_load("classification_train_dataset")

# Load label encoder from Redis
label_encoder = __conn.label_encoder_load("label-encoder")

label2id = __conn.label_encoder_load("label2id")
id2label = __conn.label_encoder_load("id2label")

encodings = __conn.label_encoder_load("classification_encodings")

# Create dataloaders with smaller batch size
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
num_labels = len(label_encoder.classes_)
tokenizer = __conn.label_encoder_load("tokenizer")

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
    )

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
classifier = IntentClassifier(model, tokenizer, label_encoder)

model = classifier.train(train_dataloader, 3, optimizer)