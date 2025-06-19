from api.loader import ModelBootstrapper

class ModelInference:
    def __init__(self):
        # Bootstrap Redis with model data from Postgres if necessary
        bootstrap = ModelBootstrapper()

        # Load the classifier and extractor models
        self.classifier = bootstrap.classifier
        self.extractor = bootstrap.extractor

    def predict_intent(self, text):
        return self.classifier.predict(text)
    
    def extract_entities(self, text):
        return self.extractor.extract_entities(text)
    
    def bot_response(self, text):
        predicted_intent = self.predict_intent(text)
        extracted_entity = self.extract_entities(text)
        return predicted_intent, extracted_entity