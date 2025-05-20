from _lib.database.redis_conn import RedisConn
import torch

redis_conn = RedisConn()

# Load model from local path
model_path = './artifact/model/intent_classifier.pth'
model = torch.load(model_path, weights_only=False)

# Save model to Redis
redis_conn.model_save(model, "intent_classifier")

# Load model from Redis
loaded_model = redis_conn.model_load("intent_classifier")
print(f"Model loaded from Redis: {loaded_model.eval()}")