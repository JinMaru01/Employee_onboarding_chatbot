from transformers import AutoTokenizer
from Intent_Classification import IntentClassifier

class DistilBertIntentModel(IntentClassifier):
    def __init__(self, model, label_encoder, tokenizer_name="distilbert-base-uncased", max_length=64):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        super().__init__(model, tokenizer, label_encoder, max_length=max_length)

    def preprocess(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
