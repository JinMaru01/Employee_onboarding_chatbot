import torch
from extend_path import sys
from _lib.database.min_io_conn import MinioConn
from transformers import DistilBertForTokenClassification, DistilBertForSequenceClassification

minio_conn = MinioConn()

# Load model from local path
classifier_path = './artifact/model/intent_classifier_v3.pth'
extractor_path = './artifact/model/entity_extractor_v3.pth'

classifier = torch.load(classifier_path, weights_only=False, map_location=torch.device('cpu'))
extractor = torch.load(extractor_path, weights_only=False, map_location=torch.device('cpu'))

# Save model to MinIO
minio_conn.upload_model(classifier, minio_conn.bucket, "intent_classifier_v3.pth")
minio_conn.upload_model(extractor, minio_conn.bucket, "extractor_v3.pth")

# Load model from MinIO
classifier_loaded = minio_conn.load_model(
    DistilBertForSequenceClassification, 
    minio_conn.bucket, 
    "intent_classifier_v3.pth",
    model_ckpt="distilbert-base-uncased", 
    num_labels=12
)