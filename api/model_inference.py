from api.loader import Loader
from _lib.database.redis_conn import RedisConn

class ModelInference():
    def __init__(self):
        self.conn = RedisConn()
        self.loader = Loader(self.conn)

    def predict_intent(self, text):
        return self.loader.classifier.predict(text)
    
    def extract_entities(self, text):
        return self.loader.extractor.extract_entities(text)
    
    def bot_response(self, text):
        predicted_intent = self.predict_intent(text)
        extracted_entity = self.extract_entities(text)
        return predicted_intent, extracted_entity