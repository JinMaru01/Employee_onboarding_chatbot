import pandas as pd
import plotly.express as px
from preprocess.data_cleaning import *

# Read raw dataset using pandas
df = pd.read_csv("./artifact/data/raw_HR_Multi_WOZ.csv")

# Drop unnecessary columns
df.drop(columns=['dialogue_id', 'turn_id'], inplace=True)

# Apply the function to the desired columns
df['speaker'] = df['speaker'].apply(convert_string_to_list_or_dict_dataframe_column)
df['utterance'] = df['utterance'].apply(convert_string_to_list_or_dict_dataframe_column)
df['state'] = df['state'].apply(convert_string_to_list_or_dict_dataframe_column)

# Extract HR and Employee messages
df["HR_message"] = df["utterance"].apply(lambda x: [msg for i, msg in enumerate(x) if i % 2 == 0])
df["Employee_message"] = df["utterance"].apply(lambda x: [msg for i, msg in enumerate(x) if i % 2 == 1])

# Display result
df[["service", "HR_message", "Employee_message"]]

df['entities'] = df['state'].apply(extract_entities)

# Call the function
entities_per_intent = get_entities_per_intent(df)
entities_all_intent = {}

# Build the dictionary with intent as key and set of entity keys as value
for intent, entities in entities_per_intent.items():
    entities_key = {key for key, value in entities}  # use set comprehension to collect keys
    entities_all_intent[intent] = entities_key

# group df by column service
grouped = df.groupby('service')

# Create a dictionary of DataFrames (one per group)
intent_dfs = {intent: service for intent, service in grouped}

# create new list of dataframe for all intent
cleaned_df = []
for intent in intent_counts.keys():
    cleaned_df.append(update_messages_df(intent_dfs, intent))

# Save each DataFrame in the list to a CSV file with a unique filename
for i, df in enumerate(cleaned_df):
    filename = f"./artifact/data/dataframe_{df['service'].iloc[0]}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {filename}")

df_full = pd.concat([cleaned_df[0], cleaned_df[1], cleaned_df[2], cleaned_df[3], cleaned_df[4],
                      cleaned_df[5], cleaned_df[6], cleaned_df[7], cleaned_df[8], cleaned_df[9]])
df_full.drop(columns=["speaker", "utterance", "state"], inplace=True)

df_full.to_csv("./artifact/data/combine_df.csv", index=False)