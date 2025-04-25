import json
import pandas as pd
from extended_function import extract_service_entities, build_ner_samples, build_ner_samples_all_intents

df = pd.read_csv("./artifact/data/combine_df.csv")

result = extract_service_entities(df)

all_results = [extract_service_entities(df, i) for i in range(len(df))]

with open('./artifact/data/entities.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4, ensure_ascii=False)

columns = df['service'].unique()

with open("./artifact/data/entities.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for intent in columns:
    ner_dataset = build_ner_samples(intent, data)

    # Optional: Save to file
    with open(f"./artifact/data/{intent}_ner.json", "w", encoding="utf-8") as f:
        json.dump(ner_dataset, f, indent=4, ensure_ascii=False)

with open("./artifact/data/all_intents_ner.json", "w", encoding="utf-8") as f:
    json.dump(build_ner_samples_all_intents(data), f, indent=4, ensure_ascii=False)