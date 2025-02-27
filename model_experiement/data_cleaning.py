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

# Function to convert single-element lists to strings
def convert_list_to_str(input_list):
    if isinstance(input_list, builtins.list) and len(input_list) == 1:
        return input_list[0]
    return input_list