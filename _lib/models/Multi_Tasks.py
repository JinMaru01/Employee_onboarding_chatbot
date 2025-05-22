import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import DistilBertModel, DistilBertTokenizerFast

class MultitaskModel(nn.Module):
    def __init__(self, intent_label_count, ner_label_count):
        super(MultitaskModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        # Intent classification head (uses [CLS] token representation)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_label_count)

        # NER head (token classification)
        self.ner_classifier = nn.Linear(self.bert.config.hidden_size, ner_label_count)

    def forward(self, input_ids, attention_mask, task="multi", labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = sequence_output[:, 0]  # [CLS] token for intent classification

        results = {}

        if task in ["intent", "multi"]:
            intent_logits = self.intent_classifier(pooled_output)
            results["intent_logits"] = intent_logits

        if task in ["ner", "multi"]:
            ner_logits = self.ner_classifier(sequence_output)
            results["ner_logits"] = ner_logits

        if labels:
            results["loss"] = self.compute_loss(results, labels)

        return results

    def compute_loss(self, outputs, labels):
        loss_fct = nn.CrossEntropyLoss()
        loss = 0

        # Intent loss
        if "intent_logits" in outputs and "intent_labels" in labels:
            intent_loss = loss_fct(outputs["intent_logits"], labels["intent_labels"])
            loss += intent_loss

        # NER loss - mask padding tokens
        if "ner_logits" in outputs and "ner_labels" in labels:
            active_loss = labels["attention_mask"].view(-1) == 1
            active_logits = outputs["ner_logits"].view(-1, outputs["ner_logits"].size(-1))[active_loss]
            active_labels = labels["ner_labels"].view(-1)[active_loss]
            ner_loss = loss_fct(active_logits, active_labels)
            loss += ner_loss

        return loss

class MultitaskDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, intent_labels, ner_labels):
        self.encodings = encodings
        self.intent_labels = intent_labels
        self.ner_labels = ner_labels

        assert len(intent_labels) == len(ner_labels), "Intent and NER label lengths must match"
        assert len(intent_labels) == len(encodings['input_ids']), "Data length mismatch"

    def __getitem__(self, idx):
        item = {key: safe_tensor(val[idx]) for key, val in self.encodings.items()}
        item["intent_labels"] = safe_tensor(self.intent_labels[idx])
        item["ner_labels"] = safe_tensor(self.ner_labels[idx])
        return item

    def __len__(self):
        return len(self.intent_labels)

def safe_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    else:
        return torch.tensor(x)
    
class ModelPipeline:
    def __init__(self, intent_label_count, ner_label_count, lr=5e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.model = MultitaskModel(intent_label_count, ner_label_count).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=lr)

    def train(self, dataloader, epochs=3):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_labels = batch["intent_labels"].to(self.device)
                ner_labels = batch["ner_labels"].to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels={
                        "intent_labels": intent_labels,
                        "ner_labels": ner_labels,
                        "attention_mask": attention_mask
                    },
                    task="multi"
                )
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"[Epoch {epoch+1}] Loss: {total_loss / len(dataloader):.4f}")

    def evaluate(self, dataloader):
        self.model.eval()

        intent_preds, intent_targets = [], []
        ner_preds, ner_targets = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                intent_labels = batch["intent_labels"].to(self.device)
                ner_labels = batch["ner_labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task="multi"
                )

                # Intent
                intent_logits = outputs["intent_logits"]
                intent_pred = torch.argmax(intent_logits, dim=1)
                intent_preds.extend(intent_pred.cpu().numpy())
                intent_targets.extend(intent_labels.cpu().numpy())

                # NER
                ner_logits = outputs["ner_logits"]
                ner_pred = torch.argmax(ner_logits, dim=2)

                for i in range(input_ids.size(0)):
                    active = attention_mask[i] == 1
                    ner_preds.append(ner_pred[i][active].cpu().tolist())
                    ner_targets.append(ner_labels[i][active].cpu().tolist())

        return intent_preds, intent_targets, ner_preds, ner_targets

    def predict(self, text, id2intent, id2ner, filter_O=True):
        self.model.eval()
        encoding = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding, task="multi")
            # Intent
            intent_id = torch.argmax(outputs["intent_logits"], dim=1).item()
            intent = id2intent[intent_id]
            # NER
            ner_ids = torch.argmax(outputs["ner_logits"], dim=2).squeeze(0)
            tokens = self.tokenizer.tokenize(text)

            ner_tags = []
            for i in range(len(tokens)):
                tag = id2ner[ner_ids[i].item()]
                if filter_O and tag == "O":
                    continue
                ner_tags.append((tokens[i], tag))

        return intent, ner_tags