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
