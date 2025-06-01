import io
import redis
import torch
import joblib

from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification

class RedisConn:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    def label_ecoder_save(self, label_encoder, file_name):
        """Save the fitted LabelEncoder to Redis"""
        buffer = io.BytesIO()
        joblib.dump(label_encoder, buffer)
        self.redis_client.set(file_name, buffer.getvalue())
        print("✅ LabelEncoder saved in Redis!")

    def label_encoder_load(self, key):
        """Load the fitted LabelEncoder from Redis"""
        encoder_bytes = self.redis_client.get(key)
        if encoder_bytes is None:
            raise ValueError("❌ LabelEncoder not found in Redis!")
        
        buffer = io.BytesIO(encoder_bytes)
        label_encoder = joblib.load(buffer)
        return label_encoder

    def model_save(self, model, file_name):
        """Save model state_dict to Redis"""
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        self.redis_client.set(file_name, buffer.getvalue())
        print(f"✅ Model state saved in Redis under key '{file_name}'!")

    def classifier_load(self, model_name, num_labels=12, model_ckpt="distilbert-base-uncased"):
        model_bytes = self.redis_client.get(model_name)
        if model_bytes is None:
            raise ValueError(f"❌ Model not found in Redis under key '{model_name}'!")
        
        # Load model architecture
        model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
        
        # Load weights from buffer
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        model.eval()

        print("✅ Model successfully loaded classifier from Redis and ready for inference!")
        return model
    
    def extractor_load(self, model_name, num_labels=26, model_ckpt="distilbert-base-uncased"):
        model_bytes = self.redis_client.get(model_name)
        if model_bytes is None:
            raise ValueError(f"❌ Model not found in Redis under key '{model_name}'!")
        
         # Load model architecture
        model = AutoModelForTokenClassification.from_pretrained(model_ckpt, num_labels=num_labels)

        # Load weights from buffer
        buffer = io.BytesIO(model_bytes)
        model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
        model.eval()

        print("✅ Model successfully loaded extractor from Redis and ready for inference!")
        return model