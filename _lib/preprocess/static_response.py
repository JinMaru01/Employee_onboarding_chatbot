import json
import pandas as pd

from extended_function import *
from intent_classifier import *

df = pd.read_csv("./artifact/data/combine_df.csv")
conversation_db = extract_conversation_pairs(df)

# Function to predict intent using the BERT model
def predict_intent_bert(text):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_label = label_encoder.inverse_transform([predicted.cpu().item()])[0]
        
    # return predicted_label, confidence.cpu().item()
    return predicted_label

def generate_response(user_message, intent, entities=None):
    """Generate a response using similar conversations and custom formatting"""
    # Find similar conversations
    similar_convs = find_similar_conversation(user_message, intent, conversation_db)
    
    if not similar_convs:
        return f"I understand this is a {intent.replace('_', ' ')}. Could you please provide more details?"
    
    # Get the most similar conversation's response
    best_response = similar_convs[-1]['bot_response']
    
    # Create a structured response
    response = f"I understand your {intent.replace('_', ' ')}. "
    response += best_response + "\n\n"
    
    # Add specific requirements based on intent
    if intent == 'training_request':
        response += "Please provide:\n- Number of participants\n- Preferred dates\n- Specific topics\n- Budget constraints"
    elif intent == 'it_issue_report':
        response += "To help resolve this quickly, please share:\n- Device details\n- Error messages\n- When the issue started\n- Steps already taken"
    elif intent == 'access_request':
        response += "To process your request, I need:\n- System/application name\n- Required access level\n- Business justification\n- Manager approval"
    elif intent == 'time_off_report':
        response += "Please confirm:\n- Exact dates\n- Type of leave\n- Handover plan"
    
    # Add entity-specific responses if available
    if entities:
        try:
            entities_dict = json.loads(entities.replace("'", '"')) if isinstance(entities, str) else entities
            if 'training_topic' in entities_dict:
                response += f"\n\nI see you're interested in {entities_dict['training_topic']} training."
            if 'issue_type' in entities_dict:
                response += f"\n\nI understand you're experiencing {entities_dict['issue_type']} issues."
        except:
            pass
    
    return response

# Test the response generator
print("Advanced Response Generator Examples:")
print("-" * 70)
test_cases = [
    ("I need to arrange machine learning training for my team",  
     {'training_topic': 'machine learning', 'number_of_participants': '5'}),
    ("My laptop keeps crashing every time I open email", 
     {'issue_type': 'system_crash', 'affected_application': 'email'}),
    ("I'd like to request vacation days for next month", 
     {'leave_type': 'vacation', 'dates': 'next month'})
]

for message, entities in test_cases:
    print(f"User: {message}")
    predicted_intent = predict_intent_bert(message)
    response = generate_response(message, predicted_intent, entities)
    print(f"Bot: {response}")
    print("-" * 70)