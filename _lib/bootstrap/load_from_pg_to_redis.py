from _lib.database.postgres_conn import PostgresConn
from _lib.database.redis_conn import RedisConn

def load_all_to_redis():
    pg = PostgresConn()
    redis = RedisConn()

    # Define keys and their loader functions
    loaders = {
        "intent_classifier_v3": lambda: pg.load_classifier("intent_classifier_v3"),
        "extractor_v3": lambda: pg.load_extractor("extractor_v3"),
        "label-encoder": lambda: pg.load_artifact("label-encoder"),
        "label2id": lambda: pg.load_artifact("label2id"),
        "id2label": lambda: pg.load_artifact("id2label"),
        "ner_id2label": lambda: pg.load_artifact("ner_id2label"),
        "tokenizer": lambda: pg.load_artifact("tokenizer"),
    }

    missing_keys = []

    for key, loader_fn in loaders.items():
        if not redis.redis_client.exists(key):
            print(f"⏳ Loading missing key from Postgres: {key}")
            obj = loader_fn()
            if "classifier" in key or "extractor" in key:
                redis.model_save(obj, key)
            else:
                redis.label_ecoder_save(obj, key)
            missing_keys.append(key)

    if not missing_keys:
        print("✅ All required models and artifacts already exist in Redis.")
    else:
        print(f"✅ Loaded {len(missing_keys)} missing item(s) from PostgreSQL to Redis: {missing_keys}")

# Allow manual run as script
if __name__ == "__main__":
    load_all_to_redis()