from extend_path import sys
from _lib.database.redis_conn import RedisConn
from _lib.models.Entity_Recognition import NamedEntityRecognizer

# Initial Redis Connection
redis_con = RedisConn()

model = redis_con.extractor_load("extractor")
tokenizer = redis_con.label_encoder_load("tokenizer")
id2label = redis_con.label_encoder_load("ner_id2label")

# Initial Model Classifier
extractor = NamedEntityRecognizer(model, tokenizer, id2label)

sample_text = "What's a company's mission?"
print("\nExtracted Entities:", extractor.extract_entities(sample_text))