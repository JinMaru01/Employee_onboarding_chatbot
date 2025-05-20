import json
from extend_path import sys
from _lib.preprocess.data_augmentation import paraphrase_t5

# Path to your JSON file
file_path = './artifact/data/sample_response.json'

# Open and load JSON
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

augmented_data = []

for row in data:
    new_questions = paraphrase_t5(row['question'])
    for q in new_questions:
        augmented_data.append({
            "intent": row['intent'],
            "question": q
        })

# Save to new file
with open('./artifact/data/augmented_response.json', 'w', encoding='utf-8') as out_file:
    json.dump(augmented_data, out_file, ensure_ascii=False, indent=2)

print("✅ Augmented data saved.")