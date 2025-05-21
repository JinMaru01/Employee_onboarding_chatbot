from extend_path import sys
from _lib.database.redis_conn import RedisConn
from _lib.models.Intent_Classification import IntentClassifier

# Initial Redis Connection
redis_con = RedisConn()

model = redis_con.classifier_load("intent-classifier")
tokenizer = redis_con.label_encoder_load("tokenizer")
label_encoder = redis_con.label_encoder_load("label-encoder")

# Initial Model Classifier
classifier = IntentClassifier(model, tokenizer, label_encoder)

# Test the model with sample messages
test_messages = [
    "vision at wing bank",
    "Tell me about company mission",
    "Tell me about wing mission",
    "Tell me about wing bank mission",
    "what is leave policy?",
    "if I want to leave 14 days, what leave I can apply?"
]

print("\nTesting model with sample messages:")
print("-" * 50)
for message in test_messages:
    intent, confidence = classifier.predict(message)
    print(f"Message: {message}")
    print(f"Predicted Intent: {intent}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 50)