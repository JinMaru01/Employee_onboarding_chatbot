from _lib.database.redis_conn import RedisConn
from _lib.bootstrap.load_from_pg_to_redis import load_all_to_redis
from _lib.models.Intent_Classification import IntentClassifier
from _lib.models.Entity_Recognition import NamedEntityRecognizer

class ModelBootstrapper:
    def __init__(self):
        self.redis = RedisConn()

        self.required_keys = [
            "intent_classifier_v3",
            "extractor_v3",
            "label-encoder", "label2id", "id2label",
            "ner_id2label", "tokenizer"
        ]

        self.ensure_model_loaded()

        # Load models and artifacts from Redis
        classifier_model = self.redis.classifier_load("intent_classifier_v3")
        extractor_model = self.redis.extractor_load("extractor_v3", num_labels=26)
        tokenizer = self.redis.label_encoder_load("tokenizer")
        label_encoder = self.redis.label_encoder_load("label-encoder")
        id2label = self.redis.label_encoder_load("ner_id2label")

        # Wrap models
        self.classifier = IntentClassifier(classifier_model, tokenizer, label_encoder)
        self.extractor = NamedEntityRecognizer(extractor_model, tokenizer, id2label)

    def ensure_model_loaded(self):
        missing = [key for key in self.required_keys if not self.redis.redis_client.exists(key)]

        if missing:
            print(f"⚠️ Missing Redis keys: {missing}")
            print("➡️ Restoring from PostgreSQL...")
            load_all_to_redis()
        else:
            print("✅ All required models/artifacts already in Redis.")