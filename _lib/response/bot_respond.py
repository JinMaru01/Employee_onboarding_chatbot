import re
from difflib import get_close_matches
from _lib.database.postgres_conn import PostgresConn

# Load your response knowledge base
pg_conn = PostgresConn()
knowledge_base = pg_conn.load_knowledge_base()

# Map synonyms to canonical detail levels
detail_level_synonyms = {
    "brief": ["brief", "summary", "summarize", "short"],
    "detailed": ["detailed", "detail", "long", "in-depth"]
}

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

def normalize_detail_level(raw_level):
        for canonical, synonyms in detail_level_synonyms.items():
            if raw_level in synonyms:
                return canonical
        return "detailed"  # default fallback

def get_chatbot_response(intent, entities, knowledge_base):
    intent = intent.strip().lower()

    for item in knowledge_base:
        if item["intent"].strip().lower() != intent:
            continue

        if "entities" in item and entities:
            for ent_type, ent_values in item["entities"].items():
                if ent_type in entities:
                    matched_key = entities[ent_type].lower()
                    for key, val in ent_values.items():
                        if key.lower() == matched_key:
                            if isinstance(val, dict):
                                # Normalize the detail level entity value
                                raw_level = entities.get("DETAIL_LEVEL", "detailed")
                                level = normalize_detail_level(raw_level)
                                # Return the matched detail level if exists, else fallback
                                return val.get(level, val.get("detailed", list(val.values())[0]))
                            return val

        # fallback default
        responses = item.get("responses", {})
        if isinstance(responses, dict):
            return responses.get("default", "I'm sorry, I don't have information on that.")
        elif isinstance(responses, list):
            return responses[0] if responses else "I'm sorry, I don't have information on that."
        else:
            return "I'm sorry, I don't have permission to show that information."

    return "Sorry, I couldn't understand your message. Could you please rephrase it or provide more details?"
