import json
import pandas as pd

class Encoder:
    def __init__(self):
        # self.redis = RedisConn()
        pass

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        label = df['service']
        return label
    
    def load_json(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return  data
    
    def fit_transform(self, data, encoder):
        # Prepare the data with correct column names
        label_encoder = encoder
        labels = label_encoder.fit_transform(data)
        return label_encoder