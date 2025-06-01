from torch.utils.data import DataLoader

from extend_path import sys
from _lib.database.redis_conn import RedisConn
from _lib.models.Intent_Classification import IntentClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Initialize Redis connection
redis_con = RedisConn()

label_encoder = redis_con.label_encoder_load("label-encoder")
test_dataset = redis_con.label_encoder_load("classification_test_dataset_v3")
model = redis_con.classifier_load("intent_classifier_v3")
tokenizer = redis_con.label_encoder_load("tokenizer")

batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

classifier = IntentClassifier(model, tokenizer, label_encoder)

predictions, true_labels = classifier.evaluate(test_dataloader)

# Compute metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

# Optionally decode labels
decoded_true = label_encoder.inverse_transform(true_labels)
decoded_pred = label_encoder.inverse_transform(predictions)

print("\nClassification Report:")
print(classification_report(decoded_true, decoded_pred))

print(f"\nAccuracy :  {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall   :  {recall:.4f}")
print(f"F1 Score :  {f1:.4f}")