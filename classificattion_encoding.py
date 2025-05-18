from _lib.database.redis_conn import RedisConn
from _lib.preprocess.data_loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

import torch

__conn = RedisConn()

# Load data
file_path = './artifact/data/annotated_data_final.json'
load = DataLoader(file_path)
data = load.data
messages = data['text']

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data['intent'])

# Save label encoder to Redis
__conn.label_ecoder_save(labels, "class-labels")

# Load label encoder from Redis
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
__conn.label_ecoder_save(encodings, "classification_encodings")

# Load encodings from Redis
encodings = __conn.label_encoder_load("classification_encodings")
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

# Create dataset
dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels_tensor)
dataset_size = len(dataset)
print(f"Dataset size: {dataset_size}")

# Split dataset
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Save train and test datasets to Redis
__conn.label_ecoder_save(train_dataset, "classification_train_dataset")
__conn.label_ecoder_save(test_dataset, "classification_test_dataset")

# Load train and test datasets from Redis
train_dataset = __conn.label_encoder_load("classification_train_dataset")
test_dataset = __conn.label_encoder_load("classification_test_dataset")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Save train and test datasets to disk
torch.save(train_dataset, './artifact/data/classification_train_dataset.pt')
torch.save(test_dataset, './artifact/data/classification_test_dataset.pt')