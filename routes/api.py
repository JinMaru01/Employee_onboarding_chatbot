import time
from flask import Blueprint, request, jsonify
from api.model_inference import ModelInference
from _lib.preprocess.log_history import log_user_interaction, load_chat_history, get_unique_dates, delete_chat_history_by_date
from _lib.response.bot_respond import get_chatbot_response, normalize_entity

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Create model instance
model = ModelInference()

@api_bp.route("/check", methods=["GET"])
def api_check():
    return jsonify({"message": "Chatbot API is up and running!"})

@api_bp.route("/predict_intent", methods=["POST"])
def predict_intent_api():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
    
    text = data["text"]
    
    start_time = time.time()
    label, confidence = model.predict_intent(text)
    end_time = time.time()

    bot_response = f"Intent: {label}"
    
    log_user_interaction(text, label, round(confidence, 4), bot_response)
    
    return jsonify({
        "text": text,
        "predicted_intent": label,
        "confidence": round(confidence, 4),
        "prediction_time": round(end_time - start_time, 4),
        "generated_response": bot_response if label == "ask_for_mission" else None
    })

@api_bp.route("/extract_entities", methods=["POST"])
def extract_entities_api():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]

    start_time = time.time()
    try:
        entity_list = model.extract_entities(text)
        entities = normalize_entity(entity_list)
    except Exception as e:
        return jsonify({"error": f"NER extraction failed: {str(e)}"}), 500
    end_time = time.time()

    return jsonify({
        "text": text,
        "entities": entities,
        "extraction_time": round(end_time - start_time, 4)
    })

@api_bp.route("/respond", methods=["POST"])
def chatbot_respond():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]

    try:
        start_time = time.time()
        intent_label, confidence = model.predict_intent(text)
        entity_list = model.extract_entities(text)
        entities = normalize_entity(entity_list)
        response = get_chatbot_response(intent_label, entities, confidence)
        end_time = time.time()

        log_user_interaction(text, intent_label, round(confidence, 4), response)

        return jsonify({
            "text": text,
            "predicted_intent": intent_label,
            "confidence": round(confidence, 4),
            "entities": entities,
            "response": response,
            "response_time": round(end_time - start_time, 4)
        })

    except Exception as e:
        return jsonify({"error": f"Failed to respond: {str(e)}"}), 500

@api_bp.route("/history/unique-dates")
def unique_dates_endpoint():
    try:
        dates = get_unique_dates()
        return jsonify({"status": "success", "dates": dates})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/history/<date>', methods=['GET'])
def get_chat_history(date):
    try:
        limit = int(request.args.get("limit", 50))
        limit = max(1, min(limit, 500))
        print(f"Fetching chat history for date: {date} with limit: {limit}")
        history = load_chat_history(date_str=date, limit=limit)
        return jsonify({"status": "success", "data": history})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/history/<date>', methods=['DELETE'])
def delete_chat_history(date):
    try:
        delete_chat_history_by_date(date)
        return jsonify({"status": "success", "message": f"Chat history for {date} deleted successfully."})
    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500