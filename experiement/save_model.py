from extend_path import sys
from _lib.database.redis_conn import RedisConn
import torch

redis_conn = RedisConn()

# Load model from local path
classifier_path = './artifact/model/intent_classifier_v2.pth'
extractor_path = './artifact/model/entity_extractor.pth'

classifier = torch.load(classifier_path, weights_only=False)
extractor = torch.load(extractor_path, weights_only=False)

# Save model to Redis
redis_conn.model_save(classifier, "intent_classifier_v2")
redis_conn.model_save(extractor, "extractor")


# Load model from Redis
classifier = redis_conn.classifier("intent_classifier_v2")
extractor = redis_conn.extractor_load("extractor")

print(f"Model loaded from Redis: {classifier.eval()}")
print(f"Model loaded from Redis: {extractor.eval()}")