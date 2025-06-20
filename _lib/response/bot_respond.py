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

def get_chatbot_response(intent, entities, confidence):
    intent = intent.strip().lower()
    score = confidence/100

    # 🚫 Block response if confidence is too low
    if score < 0.4:
        return (
            "I'm not quite sure I understood your question. "
            "Would you mind rephrasing it or giving me a bit more detail?"
        )

    # 🤔 Low-to-mid confidence (hesitant)
    elif score < 0.6:
        return (
            f"I think you might be asking about '{intent}', but I'm not entirely certain. "
            "Could you please confirm or rephrase just to make sure I respond accurately?"
        )

    # 😊 Mid-to-high confidence — add friendly preamble
    elif score < 0.8:
        preamble = (
            f"Thank you for your question! Based on what I understood about '{intent}', "
            "here's what I can share:"
        )
        return bot_response(intent, entities, preamble)
    elif score > 0.8:
        return bot_response(intent, entities)
    else:
        # Final fallback if intent not found at all
        return (
            "Thank you for your message. I'm having a little trouble understanding it clearly. "
            "Could you try rephrasing or providing a bit more context?"
        )
    
def bot_response(intent, entities, preamble=None):
    # ✅ Now look up the intent
    for item in knowledge_base:
        if item["intent"].strip().lower() != intent:
            continue

        # Check entity-specific responses
        if "entities" in item and entities:
            for ent_type, ent_values in item["entities"].items():
                if ent_type in entities:
                    matched_key = entities[ent_type].lower()
                    for key, val in ent_values.items():
                        if key.lower() == matched_key:
                            if isinstance(val, dict):
                                raw_level = entities.get("DETAIL_LEVEL", "detailed")
                                level = normalize_detail_level(raw_level)
                                response = val.get(level, val.get("detailed", list(val.values())[0]))
                            else:
                                response = val
                            return f"{preamble} {response}" if preamble else response

        # Default/fallback response
        responses = item.get("responses", {})
        if isinstance(responses, dict):
            response = responses.get("default", "I'm sorry, I don't have information on that topic just yet.")
        elif isinstance(responses, list):
            response = responses[0] if responses else "I'm sorry, I don't have information on that topic right now."
        else:
            response = "Apologies — I’m not able to provide that information at the moment."

        return f"{preamble} {response}" if preamble else response
    
    # ❌ If intent was not found at all
    fallback = (
        f"Currently I'm not able to responding to you about {intent}. It's the company policy I can't share to you in this time"
    )
    return fallback