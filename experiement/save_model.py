from extend_path import sys
from _lib.database.redis_conn import RedisConn
import torch

redis_conn = RedisConn()

# Load model from local path
classifier_path = './artifact/model/intent_classifier_v3.pth'
extractor_path = './artifact/model/entity_extractor_v3.pth'

classifier = torch.load(classifier_path, weights_only=False, map_location=torch.device('cpu'))
extractor = torch.load(extractor_path, weights_only=False, map_location=torch.device('cpu'))

# Save model to Redis
redis_conn.model_save(classifier, "intent_classifier_v3")
redis_conn.model_save(extractor, "extractor_v3")

# Load model from Redis
# classifier = redis_conn.classifier_load("intent_classifier_v3")
# extractor = redis_conn.extractor_load("extractor_v3")

# print(f"Model loaded from Redis: {classifier.eval()}")
# print(f"Model loaded from Redis: {extractor.eval()}")