import os
import sys
import json
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification
from _lib.database.redis_conn import RedisConn

# Initialize Redis connection
__conn = RedisConn()

# Load train and test datasets from Redis
train_dataset = __conn.label_encoder_load("train_dataset")
test_dataset = __conn.label_encoder_load("test_dataset")

# Load label encoder from Redis
label_encoder = __conn.label_encoder_load("label-encoder")

# Get label-to-id and id-to-label maps
label2id = __conn.label_encoder_load("label2id")
id2label = __conn.label_encoder_load("id2label")


# Load labels tensor from Redis
labels_tensor = __conn.label_encoder_load("labels_tensor")

# Create dataloaders with smaller batch size
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
num_labels = len(label2id)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

print("Starting training...")

# Training loop with early stopping
model.train()
patience = 2
best_loss = float('inf')
patience_counter = 0
num_epochs = 50

for epoch in range(num_epochs):
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device).long()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print every 10 batches
        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

    # Early stopping check
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

print("\nTraining complete.")