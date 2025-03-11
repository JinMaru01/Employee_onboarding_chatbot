import ast, builtins

# Function to convert value as string but represented to list or dict 
def convert_string_to_list_or_dict(data_item):
    for key, value in data_item.items():
        if isinstance(value, str) and (value.startswith("[") or value.startswith("{")):
            try:
                data_item[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # Ignore invalid conversions
    
    return data_item


# Function to attempt conversion using ast.literal_eval
def convert_string_to_list_or_dict_dataframe_column(data):
    if isinstance(data, str) and (data.startswith('[') or data.startswith('{')):
        try:
            return ast.literal_eval(data)
        except (ValueError, SyntaxError):
            return data  # if conversion fails, return the original string
    return data


# Function to convert single-element lists to strings
def convert_list_to_str(input_list):
    if isinstance(input_list, builtins.list) and len(input_list) == 1:
        return input_list[0]
    return input_list


# Function to extract entities from a column
def extract_entities(intent_data):
    entities = {}
    for entity_dict in intent_data:
        for key, value in entity_dict.items():
            entities[key] = value
    return entities


# Funciton to extract entities from intent
def get_entities_per_intent(df):
    entity_dict = {}

    for _, row in df.iterrows():
        service = row["service"]
        entities = row["entities"]

        if isinstance(entities, dict):
            if service not in entity_dict:
                entity_dict[service] = []
            entity_dict[service].extend(entities.items())  # Convert dict to list of tuples

    return entity_dict


# Funciton to update message by removing "HR Assistant" and "Enployee" in the Prefix
def update_messages_df(df, intent):
    intent_df = df[intent].copy()
    # Clean the HR_message column: remove the "HR Assistant:" prefix from every message in the list
    intent_df["HR_message"] = intent_df["HR_message"].apply(
        lambda messages: [msg.split("HR Assistant:")[-1].strip() for msg in messages]
    )
    
    # Clean the Employee_message column: remove the "Employee:" prefix from every message in the list
    intent_df["Employee_message"] = intent_df["Employee_message"].apply(
        lambda messages: [msg.split("Employee:")[-1].strip() for msg in messages]
    )
    
    return intent_df