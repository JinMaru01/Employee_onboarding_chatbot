import time
from flask import Flask, render_template, request, jsonify
from api.model_inference import ModelInference

from _lib.response.mission import mission_data, respond_to_mission_question

app = Flask(__name__, static_folder='static')

model = ModelInference()

@app.route("/", methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/api", methods=["GET"])
def api_check():
    return jsonify({"message": "Chatbot API is up and running!"})

# POST endpoint for prediction
@app.route("/api/predict_intent", methods=["POST"])
def predict_intent():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]

    start_time = time.time()
    label, confidence = model.predict_intent(text)
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
