import abc
import io
import os
import torch
import joblib
import redis
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification

# Base Interface
class BaseModelStore(abc.ABC):
    @abc.abstractmethod
    def classifier_load(self, model_name, **kwargs):
        pass

    @abc.abstractmethod
    def extractor_load(self, model_name, **kwargs):
        pass

    @abc.abstractmethod
    def label_encoder_load(self, key):
        pass

# Redis Implementation
class RedisConn(BaseModelStore):
    def __init__(self):
        load_dotenv()
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, db=redis_db)

    def label_encoder_load(self, key):
        encoder_bytes = self.redis_client.get(key)
        if encoder_bytes is None:
            raise ValueError("LabelEncoder not found in Redis!")
        return joblib.load(io.BytesIO(encoder_bytes))

    def classifier_load(self, model_name, num_labels=12, model_ckpt="distilbert-base-uncased"):
        model_bytes = self.redis_client.get(model_name)
        if model_bytes is None:
            raise ValueError(f"Model '{model_name}' not found in Redis!")
        model = DistilBertForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        model.eval()
        return model

    def extractor_load(self, model_name, num_labels=26, model_ckpt="distilbert-base-uncased"):
        model_bytes = self.redis_client.get(model_name)
        if model_bytes is None:
            raise ValueError(f"Model '{model_name}' not found in Redis!")
        model = DistilBertForTokenClassification.from_pretrained(model_ckpt, num_labels=num_labels)
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        model.eval()
        return model

# MinIO Implementation
class MinioConn(BaseModelStore):
    def __init__(self):
        load_dotenv()
        self.bucket = os.getenv("MINIO_BUCKET", "models")
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
            secure=False
        )

    def label_encoder_load(self, key):
        try:
            response = self.client.get_object(self.bucket, key)
            return joblib.load(io.BytesIO(response.read()))
        except S3Error as e:
            print(f"Failed to load label encoder '{key}': {e}")
            return None

    def classifier_load(self, model_name, num_labels=12, model_ckpt="distilbert-base-uncased"):
        try:
            response = self.client.get_object(self.bucket, model_name)
            buffer = io.BytesIO(response.read())
            model = DistilBertForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
            model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
            model.eval()
            return model
        except S3Error as e:
            print(f"Failed to load classifier model '{model_name}': {e}")
            return None

    def extractor_load(self, model_name, num_labels=26, model_ckpt="distilbert-base-uncased"):
        try:
            response = self.client.get_object(self.bucket, model_name)
            buffer = io.BytesIO(response.read())
            model = DistilBertForTokenClassification.from_pretrained(model_ckpt, num_labels=num_labels)
            model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
            model.eval()
            return model
        except S3Error as e:
            print(f"Failed to load extractor model '{model_name}': {e}")
            return None
