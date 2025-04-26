import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_message(msg):
    if isinstance(msg, str):
        msg = msg.strip('[]').strip("'").strip('"')
        if '\", \"' in msg:
            msg = msg.split('\", \"')[0]
        elif "', '" in msg:
            msg = msg.split("', '")[0]
    return msg

# Create conversation database
def extract_conversation_pairs(df):
    conversations = []
    for _, row in df.iterrows():
        hr_message = clean_message(row['HR_message'])
        employee_message = clean_message(row['Employee_message'])
        service = row['service']
        entities = row['entities']
        conversations.append({
            'intent': service,
            'user_message': employee_message,
            'bot_response': hr_message,
            'entities': entities
        })
    return conversations

def find_similar_conversation(user_message, intent, conversations, top_k=3):
    """Find similar conversations based on user message and intent"""
    # Filter conversations by intent
    intent_conversations = [conv for conv in conversations if conv['intent'] == intent]
    if not intent_conversations:
        return None
    
    # Create vectors for similarity comparison
    messages = [conv['user_message'] for conv in intent_conversations]
    vectorizer = TfidfVectorizer()
    message_vectors = vectorizer.fit_transform(messages)
    query_vector = vectorizer.transform([user_message])
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, message_vectors)[0]
    top_indices = np.argsort(similarities)[-top_k:]
    
    return [intent_conversations[i] for i in top_indices]

def extract_service_entities(df, row_index=0):
    service = df['service'][row_index]
    json_str = df['entities'][row_index]

    try:
        json_obj = ast.literal_eval(json_str)
        return {service: json_obj}
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing JSON string at row {row_index}: {e}")
        return {service: None}

def convert_to_ner_format(item, intent_key="training_request"):
    ner_entry = {}
    entities = []
    
    # Construct the sentence using the values (basic example)
    request = item[intent_key]
    sentence_parts = [f"{key.replace('_', ' ')}: {value}" for key, value in request.items()]
    sentence = ". ".join(sentence_parts) + "."

    for key, value in request.items():
        start = sentence.find(value)
        if start != -1:
            end = start + len(value)
            entities.append({"start": start, "end": end, "label": key})
    
    ner_entry["content"] = sentence
    ner_entry["entities"] = entities
    return ner_entry

def build_ner_samples(intent_name, data):
    ner_data = []

    for record in data:
        # Ensure the intent exists and is a dictionary
        if intent_name not in record or not isinstance(record[intent_name], dict):
            continue

        fields = record[intent_name]  # this is a dict of key-value pairs
        sentence_parts = [f"{k.replace('_', ' ')}: {v}" for k, v in fields.items()]
        sentence = ". ".join(sentence_parts) + "."

        entities = []
        for label, value in fields.items():
            start = sentence.find(value)
            if start != -1:
                end = start + len(value)
                entities.append({
                    "start": start,
                    "end": end,
                    "label": label
                })

        ner_data.append({
            "content": sentence,
            "entities": entities
        })

    return ner_data

def build_ner_samples_all_intents(data):
    merged_ner_data = []

    for record in data:
        for intent_name, fields in record.items():
            if not isinstance(fields, dict):
                continue

            # Build the sentence from key-value pairs
            sentence_parts = []
            value_positions = []
            for k, v in fields.items():
                part = f"{k.replace('_', ' ')}: {v}"
                sentence_parts.append(part)
                value_positions.append((k, v))  # Store for span indexing

            sentence = ". ".join(sentence_parts) + "."

            # Find positions of values in the sentence (avoid duplicated value collisions)
            entities = []
            cursor = 0
            for label, value in value_positions:
                try:
                    # Look for value in the sentence starting from `cursor` to avoid overlap problems
                    start = sentence.index(value, cursor)
                    end = start + len(value)
                    entities.append({
                        "start": start,
                        "end": end,
                        "label": label
                    })
                    cursor = end  # Move cursor forward
                except ValueError:
                    continue  # Value not found, skip

            merged_ner_data.append({
                "content": sentence,
                "entities": entities,
                "intent": intent_name
            })

    return merged_ner_data