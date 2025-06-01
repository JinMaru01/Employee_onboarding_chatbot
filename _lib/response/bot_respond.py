import re
from difflib import get_close_matches
import json

# Load your response knowledge base
with open("./artifact/data/json/response_design.json", "r") as f:
    knowledge_base = json.load(f)
    
mission_aliases = {
    "customer": "Customer at the heart",
    "community": "Community as the cause",
    "employee": "Employee as the pillar",
    "shareholder": "Deliver shareholder value",
    "mission 1": "Customer at the heart",
    "mission 2": "Community as the cause",
    "mission 3": "Employee as the pillar",
    "mission 4": "Deliver shareholder value",
    "first mission": "Customer at the heart",
    "second mission": "Community as the cause",
    "third mission": "Employee as the pillar",
    "fourth mission": "Deliver shareholder value"
}

mission_data = {
    "message": "The mission of the company is",
    "response": {
        "Customer at the heart": "To engage and understand customer need, provide best-in-class product and Services, be respective and quick in resolving-resulting in true customer delight.",
        "Community as the cause": "To deliver robust and cost-effective mobile money service that promote financial inclusion, catalyze growth, and reduce social inequality.",
        "Employee as the pillar": "To provide an enabling work culture, where career aspirations can be realized through consistent performance and demonstration of WING's core values.",
        "Deliver shareholder value": "To demonstrate strong corporate governance standards that protects and balance shareholder interests in the journey to achieving short and long-term business goals and staying relevant by being the first bank to offer innovate mobile financial service, accessible to all."
    }
}

def respond_to_mission_question(intent: str, question: str, mission_data: dict) -> str:
    message = mission_data.get("message", "The mission of the company is")
    missions = mission_data.get("response", {})
    mission_titles = list(missions.keys())
    user_question = question.lower()
        
    match = re.search(r"mission\s*(\d+)", user_question)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(mission_titles):
            return missions[mission_titles[idx]]
        else:
            return f"There is no mission number {match.group(1)}. Please ask about mission 1 to {len(mission_titles)}."

    # Alias keyword matching
    for keyword, title in mission_aliases.items():
        if keyword in user_question:
            return missions.get(title)

    # Fuzzy match using difflib
    for title in mission_titles:
        if get_close_matches(title.lower(), [user_question], n=1, cutoff=0.6):
            return missions[title]

    # Asking for all missions
    if any(p in user_question for p in ["all missions", "each mission", "full list"]):
        return "\n\n".join([f"**{t}**: {d}" for t, d in missions.items()])

    # General intent fallback
    if intent == "ask_for_mission":
        return f"{message} {', '.join(mission_titles)}."

    return "Could you clarify which mission you're referring to?"

def normalize_entity(entities):
    # Normalize model entities output to usable dict
    normalized_entities = {}
    for ent in entities:
        key = ent["type"]
        val = ent["entity"].strip().lower()
        if val not in ["[sep]", "[cls]", "[pad]"]:  
            if key not in normalized_entities:
                normalized_entities[key] = val

    return normalized_entities

def get_chatbot_response(intent, entities, knowledge_base):
    intent = intent.strip().lower()

    for item in knowledge_base:
        if item["intent"].strip().lower() != intent:
            continue

        # Entity-based match
        if "entities" in item and entities:
            for ent_type, ent_values in item["entities"].items():
                if ent_type in entities:
                    matched_key = entities[ent_type].lower()
                    for key, val in ent_values.items():
                        if key.lower() == matched_key:
                            # Nested (with DETAIL_LEVEL)
                            if isinstance(val, dict):
                                level = entities.get("DETAIL_LEVEL", "detailed").lower()
                                return val.get(level, val.get("detailed", list(val.values())[0]))
                            return val

        # Return default fallback for this intent
        responses = item.get("responses", {})
        if isinstance(responses, dict):
            return responses.get("default", "I'm sorry, I don't have information on that.")
        elif isinstance(responses, list):
            return responses[0] if responses else "I'm sorry, I don't have information on that."
        else:
            return "I'm sorry, I don't have permission to show on that information."

    return "Sorry, I couldn't understand your message. Could you please rephrase it or provide more details?"