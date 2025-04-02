import torch

from connection.redis_conn import RedisConn
from preprocess.label_encoder import Encoder

from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder


__conn = RedisConn()
__encoder = Encoder()

# Read dataset
file_path = "../data/combine_df.csv"
label_encoder = __encoder.fit_transform(__encoder.load_data(file_path), LabelEncoder())

# Load Label encoder from Redis
label_encoder = __conn.save_label_encoder(label_encoder, "label-encoder")

# Load model from local and store in the redis
# model = torch.load("../model/model_distilbert_25epochs.pth", weights_only=False)
# _conn.model_save(model, "distilbert_state")

# Load model and encoder from redis
model = __conn.model_load("distilbert_state")
label_encoder = __conn.load_label_encoder("label-encoder")


# Initial tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
max_length = 64

# Function to predict intent using the BERT model
def predict_intent(text):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = label_encoder.inverse_transform([predicted.cpu().item()])[0]
        
    return predicted_label, confidence.cpu().item()

# Test the model with sample messages
test_messages = [
    "I need to request time off for next week",
    "My computer keeps crashing and I can't work",
    "I want to set up training for my team",
    "I need access to the sales database",
    "I want to report the harm",
    "I would like to sign up for a training session on effective communication.",
    "Could you please do my performance review meeting?",
    "I need access to the shared project drive; can you help me out?",
    "I’m planning to relocate to the New York office; what is the process?",
    "I want to report a safety incident that occurred in the warehouse.",
    "I’d like to request time off for next month due to a personal commitment.",
    "Can you help me enroll in the company benefits program, specifically health insurance?",
    "I need to report an incident of harassment that I witnessed at work.",
    "How do I set my performance goals for the next quarter?",
    # "My computer isn’t booting up properly; can you assist me with this issue?"
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\nTesting model with sample messages:")
print("-" * 50)
for message in test_messages:
    intent, confidence = predict_intent(message)
    print(f"Message: {message}")
    print(f"Predicted Intent: {intent}")
    print(f"Confidence: {confidence:.2f}")
    print("-" * 50)