from extend_path import sys
from torch.utils.data import DataLoader
from _lib.database.redis_conn import RedisConn
from _lib.models.Entity_Recognition import NamedEntityRecognizer
from seqeval.scheme import IOB2
from seqeval.metrics import (
    f1_score, 
    recall_score, 
    accuracy_score,
    precision_score, 
    classification_report
    )

# Initialize Redis connection
redis_con = RedisConn()

test_dataset = redis_con.label_encoder_load("ner_test_dataset_v3")
model = redis_con.extractor_load("extractor_v3")
tokenizer = redis_con.label_encoder_load("tokenizer")
id2label = redis_con.label_encoder_load("ner_id2label")

batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

extractor = NamedEntityRecognizer(model, tokenizer, id2label)

pred_labels, true_labels = extractor.evaluate(test_dataloader)

# Compute metrics
print(classification_report(true_labels, pred_labels, mode='strict', scheme=IOB2))
print(f"Accuracy: {accuracy_score(true_labels, pred_labels):.4f}")
print(f"F1 Score: {f1_score(true_labels, pred_labels):.4f}")
print(f"Precision: {precision_score(true_labels, pred_labels):.4f}")
print(f"Recall: {recall_score(true_labels, pred_labels):.4f}")