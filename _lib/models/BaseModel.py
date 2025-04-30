import torch
from Intent_Classification import IntentClassifier

class BaseModel():
    def __init__(self, model, tokenizer, label_encoder, device=None, max_length=64):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
