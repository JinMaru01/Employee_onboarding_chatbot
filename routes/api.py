import time
from flask import Blueprint, request, jsonify
from api.model_inference import ModelInference
from _lib.preprocess.log_history import log_user_interaction
from _lib.response.mission import mission_data, respond_to_mission_question

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Create model instance
model = ModelInference()

@api_bp.route("", methods=["GET"])
def api_check():
    """
    Check if API is running.
    
    Returns:
        JSON response with status message
    """
    return jsonify({"message": "Chatbot API is up and running!"})

@api_bp.route("/predict_intent", methods=["POST"])
def predict_intent():
    """
    Predict intent from user text.
    
    Returns:
        JSON response with prediction results
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    
    text = data["text"]
    
    start_time = time.time()
    label, confidence = model.predict_intent(text)
    end_time = time.time()
    
    # Default response
    bot_response = f"Intent: {label}"
    
    if label == "ask_for_mission":
        bot_response = respond_to_mission_question(label, text, mission_data)
    
    # Log interaction
    log_user_interaction(text, label, round(confidence, 4), bot_response)
    
    return jsonify({
        "text": text,
        "predicted_intent": label,
        "confidence": round(confidence, 4),
        "prediction_time": round(end_time - start_time, 4),
        "generated_response": bot_response if label == "ask_for_mission" else None
    })