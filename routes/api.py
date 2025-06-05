import time
from flask import Blueprint, request, jsonify
from api.model_inference import ModelInference
from _lib.preprocess.log_history import log_user_interaction
from _lib.response.bot_respond import knowledge_base, respond_to_mission_question, get_chatbot_response, normalize_entity

api_bp = Blueprint('api', __name__, url_prefix='/api')

# Create model instance
model = ModelInference()

@api_bp.route("/check", methods=["GET"])
def api_check():
    """
    Check if API is running.
    
    Returns:
        JSON response with status message
    """
    return jsonify({"message": "Chatbot API is up and running!"})

@api_bp.route("/predict_intent", methods=["POST"])
def predict_intent_api():
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

@api_bp.route("/extract_entities", methods=["POST"])
def extract_entities_api():
    """
    Extract entities from user input text using the NER model.

    Returns:
        JSON response with extracted entities
    """
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
    """
    Generate chatbot response using predicted intent and extracted entities.

    Returns:
        JSON response with predicted intent, entities, and chatbot response
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data["text"]

    try:
        start_time = time.time()

        # Predict intent
        intent_label, confidence = model.predict_intent(text)

        # Extract entities
        entity_list = model.extract_entities(text)
        entities = normalize_entity(entity_list)

        # Get chatbot response from KB
        response = get_chatbot_response(intent_label, entities, knowledge_base)

        end_time = time.time()

        # Log the interaction
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
