from _lib.models.Intent_Classification import IntentClassifier
from _lib.database.redis_conn import RedisConn

class ModelLoader():
    def __init__(self):
        redis_con = RedisConn()
        self.model = redis_con.model_load("intent-classifier")
        self.tokenizer = redis_con.label_encoder_load("tokenizer")
        self.label_encoder = redis_con.label_encoder_load("label-encoder")

        # Initial Model Classifier
        self.classifier = IntentClassifier(self.model, self.tokenizer, self.label_encoder)