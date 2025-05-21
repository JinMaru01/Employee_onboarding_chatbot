import torch
import torch.nn as nn

class IntentClassifier():
    def __init__(self, model, tokenizer, label_encoder, device=None, max_length=64):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess(self, text):
        return  self.tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=64
    )

    def train(self, trainset, optimizer, num_epochs, use_custom_loss=False, early_stopping_patience=2):
        self.model.train()

        # Initial training parameters
        best_loss = float('inf')
        patience_counter = 0
        criterion = nn.CrossEntropyLoss()

        print("Starting training...")

        for epoch in range(num_epochs):
            epoch_loss = 0

            for i, batch in enumerate(trainset):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device).long()

                optimizer.zero_grad()

                if use_custom_loss:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Print every 10 batches
                if (i + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {i+1}/{len(trainset)}, Loss: {loss.item():.4f}")

            avg_epoch_loss = epoch_loss / len(trainset)
            print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break

        print("\nTraining complete.")
        return  self.model
    
    def evaluate(self, testset):
        # Evaluation
        print("\nEvaluating model...")
        self.model.eval()
        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in testset:
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, dim=1)

                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        return  predictions, true_labels
    
    def predict(self, text):
        inputs = self.preprocess(text)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            label = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
            confidence_score = confidence.cpu().item()

        return label, confidence_score*100