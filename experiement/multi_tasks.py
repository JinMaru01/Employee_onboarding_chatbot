from extend_path import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from _lib.database.redis_conn import RedisConn
from _lib.preprocess.preparation import prepare_multitask_data
from _lib.models.Multi_Tasks import ModelPipeline

# connect to redis
redis_con = RedisConn()

json_path = "./artifact/data/json/annotated_data.json"

with open(json_path, 'r') as file:
    raw_data = json.load(file)

# Split the dataset (e.g., 80% train, 20% test)
train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

# Now prepare datasets separately
train_dataset, intent_encoder, tag_encoder = prepare_multitask_data(train_data)
test_dataset, _, _ = prepare_multitask_data(test_data)

# Save encoders to Redis
redis_con.label_ecoder_save(intent_encoder.classes_.tolist(), "intent_labels_multi_task")
redis_con.label_ecoder_save(tag_encoder.classes_.tolist(), "ner_tags_multi_task")
redis_con.label_ecoder_save(intent_encoder, "intent_label_encoder_multi_task")
redis_con.label_ecoder_save(tag_encoder, "ner_label_encoder_multi_task")

# Save label2id and id2label mappings
label2id_intent = {label: idx for idx, label in enumerate(intent_encoder.classes_)}
label2id_ner = {label: idx for idx, label in enumerate(tag_encoder.classes_)}
id2label_intent = {idx: label for label, idx in label2id_intent.items()}
id2label_ner = {idx: label for label, idx in label2id_ner.items()}

redis_con.label_ecoder_save(label2id_intent, "intent_label2id_multi_task")
redis_con.label_ecoder_save(id2label_intent, "intent_id2label_multi_task")

# save data to redis
redis_con.label_ecoder_save(label2id_ner, "ner_label2id_multi_task")
redis_con.label_ecoder_save(id2label_ner, "ner_id2label_multi_task")

# Save train and test datasets to Redis
redis_con.label_ecoder_save(train_dataset, "multi_task_train_dataset")
redis_con.label_ecoder_save(test_dataset, "multi_task_test_dataset")

# Load encoders from Redis
intent_encoder = redis_con.label_encoder_load("intent_label_encoder_multi_task")
tag_encoder = redis_con.label_encoder_load("ner_label_encoder_multi_task")

# Save data to as pt tensors
torch.save(train_dataset, "./artifact/data/train/multi_tasks.pt")
torch.save(test_dataset, "./artifact/data/test/multi_tasks.pt")

# Load label2id and id2label mappings from Redis
label2id_intent = redis_con.label_encoder_load("intent_label2id_multi_task")
label2id_ner = redis_con.label_encoder_load("ner_label2id_multi_task")
id2label_intent = redis_con.label_encoder_load("intent_id2label_multi_task")
id2label_ner = redis_con.label_encoder_load("ner_id2label_multi_task")

# Load train dataset from Redis
train_dataset = redis_con.label_encoder_load("multi_task_train_dataset")
# Load test dataset from Redis
test_dataset = redis_con.label_encoder_load("multi_task_test_dataset")

# Get the number of labels
intent_label_count = len(intent_encoder.classes_)
ner_label_count = len(tag_encoder.classes_)

# Get ID to label mappings (for decoding during inference)
id2intent = {i: label for i, label in enumerate(intent_encoder.classes_)}
id2ner = {i: label for i, label in enumerate(tag_encoder.classes_)}

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
pipeline = ModelPipeline(intent_label_count=len(id2intent), ner_label_count=len(id2ner))

pipeline.train(train_loader, epochs=50)

model_path = "./artifact/model/multi_task_v3.pth"
torch.save(pipeline.model, model_path)