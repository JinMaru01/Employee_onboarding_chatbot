import io
import redis
import torch
import joblib

from transformers import DistilBertForSequenceClassification

class RedisConn:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def model_save(self, model, file_name):
        """Save model state_dict to Redis"""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        self.redis_client.set(file_name, buffer.getvalue())
        print("✅ Model state saved in Redis!")

    def model_load(self, model_name):
        """Load model state_dict from Redis without saving locally"""
        model_bytes = self.redis_client.get(model_name)
        if model_bytes is None:
            raise ValueError("❌ Model not found in Redis!")
        
        buffer = io.BytesIO(model_bytes)
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=20)
        model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        model.eval()
        print("✅ Model successfully loaded from Redis and ready for inference!")
        return model

    def save_label_encoder(self, label_encoder, file_name):
        """Save the fitted LabelEncoder to Redis"""
        buffer = io.BytesIO()
        joblib.dump(label_encoder, buffer)
        self.redis_client.set(file_name, buffer.getvalue())
        print("✅ LabelEncoder saved in Redis!")

    def load_label_encoder(self, key):
        """Load the fitted LabelEncoder from Redis"""
        encoder_bytes = self.redis_client.get(key)
        if encoder_bytes is None:
            raise ValueError("❌ LabelEncoder not found in Redis!")
        
        buffer = io.BytesIO(encoder_bytes)
        label_encoder = joblib.load(buffer)
        print("label from load", label_encoder)
        print("✅ LabelEncoder successfully loaded from Redis!")
        return label_encoder