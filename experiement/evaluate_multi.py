import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_report
from extend_path import sys
from _lib.database.redis_conn import RedisConn
from _lib.models.Multi_Tasks import ModelPipeline

redis_con = RedisConn()

intent_encoder = redis_con.label_encoder_load("intent_label_encoder_multi_task")
tag_encoder = redis_con.label_encoder_load("ner_label_encoder_multi_task")

# Get the number of labels
intent_label_count = len(intent_encoder.classes_)
ner_label_count = len(tag_encoder.classes_)

# Get ID to label mappings (for decoding during inference)
id2intent = {i: label for i, label in enumerate(intent_encoder.classes_)}
id2ner = {i: label for i, label in enumerate(tag_encoder.classes_)}

# load datasets and models from Redis
test_dataset = redis_con.label_encoder_load("multi_task_test_dataset")

test_loader = DataLoader(test_dataset, batch_size=32)

model = torch.load("./artifact/model/multi_task_v3.pth", weights_only=False, map_location=torch.device('cpu'))

# Initialize the model pipeline
pipeline = ModelPipeline(intent_label_count=len(id2intent), ner_label_count=len(id2ner))

# Run evaluation after training
intent_preds, intent_labels, ner_preds, ner_labels = pipeline.evaluate(test_loader, model=model)

# Decode predictions
true_intents = intent_encoder.inverse_transform(intent_labels)
pred_intents = intent_encoder.inverse_transform(intent_preds)

# Decode NER labels
true_tags = [[tag_encoder.inverse_transform([t])[0] for t in seq if t != -100] for seq in ner_labels]
pred_tags = [[tag_encoder.inverse_transform([t])[0] for t, gt in zip(seq, gts) if gt != -100] for seq, gts in zip(ner_preds, ner_labels)]

# Intent evaluation
print("Intent Classification Report:\n", classification_report(true_intents, pred_intents))

# NER evaluation
print("NER Tagging Report:\n", seqeval_report(true_tags, pred_tags))
