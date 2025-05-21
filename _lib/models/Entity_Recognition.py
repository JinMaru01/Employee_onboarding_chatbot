import torch
import torch.nn as nn

class NamedEntityRecognizer:
    def __init__(self, model, tokenizer, id2label, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.num_labels = len(id2label)
        
        self.model = model
        self.model.to(self.device)

    def train(self, dataloader, optimizer, num_epochs=5, use_custom_loss=False, early_stopping_patience=2):
        self.model.train()
        
        # Initial training parameters
        best_loss = float('inf')
        patience_counter = 0
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        for epoch in range(num_epochs):
            total_loss = 0.0

            for i, batch in enumerate(dataloader):
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                optimizer.zero_grad()

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                if use_custom_loss:
                    logits = outputs.logits.view(-1, self.model.config.num_labels)
                    labels_flat = labels.view(-1)
                    loss = criterion(logits, labels_flat)
                else:
                    loss = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("🛑 Early stopping triggered.")
                    break

        self.model.eval()
        return self.model


    def evaluate(self, dataloader):
        self.model.eval()
        true_labels = []
        pred_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=2)

                for i in range(labels.size(0)):
                    true_seq = []
                    pred_seq = []
                    for j in range(labels.size(1)):
                        if labels[i][j] == -100:
                            continue
                        true_seq.append(self.id2label[labels[i][j].item()])
                        pred_seq.append(self.id2label[predictions[i][j].item()])
                    true_labels.append(true_seq)
                    pred_labels.append(pred_seq)
        return pred_labels, true_labels

    def extract_entities(self, text):
        self.model.eval()
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)

        predicted_labels = [self.id2label[p.item()] for p in predictions[0]]
        tokens_decoded = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        entities = []
        current_entity = ""
        current_label = None

        for token, label in zip(tokens_decoded, predicted_labels):
            if label.startswith("B-"):
                if current_entity:
                    entities.append({"entity": current_entity, "type": current_label})
                current_entity = token
                current_label = label[2:]
            elif label.startswith("I-") and current_entity:
                current_entity += token.replace("##", "")
            else:
                if current_entity:
                    entities.append({"entity": current_entity, "type": current_label})
                    current_entity = ""
                    current_label = None

        if current_entity:
            entities.append({"entity": current_entity, "type": current_label})

        return entities