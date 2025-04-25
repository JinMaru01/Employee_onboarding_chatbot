import torch
import numpy as np
import pandas as pd

from preprocess.extended_function import clean_message
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, DistilBertForSequenceClassification

df = pd.read_csv("./artifact/data/combine_df.csv")

# Initialize tokenizer and prepare data
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Prepare the data with correct column names
messages = df['Employee_message'].apply(clean_message)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(df['service'])

# Tokenize with smaller max_length
max_length = 64
encodings = tokenizer(
    messages.tolist(),
    truncation=True,
    padding=True,
    max_length=max_length,
    return_tensors='pt'
)

# Create dataset
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']
labels_tensor = torch.tensor(labels)
dataset = TensorDataset(input_ids, attention_mask, labels_tensor)

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create dataloaders with smaller batch size
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model
num_labels = len(label_encoder.classes_)

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
num_epochs = 25

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

# Save the whole model
print("\nSaving model...")
torch.save(model, "../model/model_25epochs.pth")

# Save the model's state dictionary
torch.save(model.state_dict(), "../model/model_state_25epochs.pth")
print("\nSaving complete.")

# Evaluation
print("\nEvaluating model...")
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

accuracy = np.mean(np.array(predictions) == np.array(true_labels))
print(f"Test Accuracy: {accuracy:.4f}")