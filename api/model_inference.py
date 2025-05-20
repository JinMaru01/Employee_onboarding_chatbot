from api.model_loader import ModelLoader

class ModelInference():
    def __init__(self):
        self.loader = ModelLoader()

    def predict_intent(self, text):
        return self.loader.classifier.predict(text)