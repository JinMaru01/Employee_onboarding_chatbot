import torch

class IntentClassifier():
    def __init__(self, model, tokenizer, label_encoder, device=None, max_length=64):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        inputs = self.preprocess(text)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            label = self.label_encoder.inverse_transform([predicted.cpu().item()])[0]

        return label, confidence.cpu().item()