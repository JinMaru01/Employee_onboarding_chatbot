import re
import string
import pandas as pd
from transformers import AutoTokenizer

# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

data = pd.read_json("./artifact/data/ner_dataset/entities.json")

result = '\n'.join(
    str(v) 
    for row in data.values 
    for cell in row 
    if isinstance(cell, dict) 
    for v in cell.values()
)

with open('./artifact/data/ner_dataset/tokens_output4.txt', 'w', encoding='utf-8') as f:
    f.write(str(result) + ' ')

def remove_punctuation_using_regex(input_string):
    regex_pattern = f"[{re.escape(string.punctuation)}]"
    result = re.sub(regex_pattern, "", input_string)
    return result

tokens = tokenizer.tokenize(result)

with open('./artifact/data/ner_dataset/tokens_output2.txt', 'w', encoding='utf-8') as f:
    for token in tokens:
        f.write(str(token) + ' ')