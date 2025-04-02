import pandas as pd

class Encoder:
    def __init__(self):
        # self.redis = RedisConn()
        pass

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        label = df['service']
        return label
    
    def fit_transform(self, data, encoder):
        # Prepare the data with correct column names
        label_encoder = encoder
        labels = label_encoder.fit_transform(data)
        return label_encoder