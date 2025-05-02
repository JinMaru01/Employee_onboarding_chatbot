import time
import torch
from flask_cors import CORS
from transformers import AutoTokenizer
from flask import Flask, render_template, request, jsonify
from _lib.response.mission import mission_data, respond_to_mission_question

from _lib.database.redis_conn import RedisConn
from _lib.preprocess.label_encoder import Encoder

app = Flask(__name__, static_folder='static')
CORS(app)

# Load model, tokenizer, and label encoder
__conn = RedisConn()
__encoder = Encoder()
model = __conn.model_load("distilbert_state_21")
label_encoder = __conn.load_label_encoder("label-encoder-new")

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
max_length = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_intent(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        label = label_encoder.inverse_transform([predicted.cpu().item()])[0]

    return label, confidence.cpu().item()

@app.route("/api", methods=["GET"])
def api_check():
    return jsonify({"message": "Chatbot API is up and running!"})

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

# POST endpoint for prediction
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]

    start_time = time.time()
    label, confidence = predict_intent(text)
    end_time = time.time()

    response_data = {
        "text": text,
        "predicted_intent": label,
        "confidence": round(confidence, 4),
        "prediction_time": round(end_time - start_time, 4)
    }

    # If the intent is about company mission, generate a mission response
    if label == "ask_for_mission":
        mission_response = respond_to_mission_question(label, text, mission_data)
        response_data["generated_response"] = mission_response
    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True)
