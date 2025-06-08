from _lib.models.Intent_Classification import IntentClassifier
from _lib.models.Entity_Recognition import NamedEntityRecognizer
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification

class MinioLoader:
    def __init__(self, minio_conn):

        # --- Load Components ---
        # Intent Classifier
        classifier_model = minio_conn.load_model(
            model_class=DistilBertForSequenceClassification,
            bucket_name=minio_conn.bucket,
            object_name="intent_classifier_v3.pth",
            model_ckpt="distilbert-base-uncased",
            num_labels=12
        )
        tokenizer = minio_conn.label_encoder_load("tokenizer.joblib")
        label_encoder = minio_conn.label_encoder_load("label-encoder.joblib")

        self.classifier = IntentClassifier(classifier_model, tokenizer, label_encoder)

        # NER Extractor
        extractor_model = minio_conn.load_model(
            model_class=DistilBertForTokenClassification,
            bucket_name=minio_conn.bucket,
            object_name="extractor_v3.pth",
            model_ckpt="distilbert-base-uncased",
            num_labels=26
        )
        id2label = minio_conn.label_encoder_load("ner_id2label.joblib")

        self.extractor = NamedEntityRecognizer(extractor_model, tokenizer, id2label)
