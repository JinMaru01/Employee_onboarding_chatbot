import json
import pandas as pd

class DataLoader:
     def __init__(self, file_path: str):
          self.file_path = file_path
          self.file_extension = file_path.split('.')[-1].lower()
          print(f"Loading data from {file_path} with extension {self.file_extension}")
          if self.file_extension not in ['csv', 'json']:
               raise ValueError("Unsupported file format. Please use 'csv' or 'json'.")
          
          # Load data based on file extension
          if self.file_extension == 'csv':
               self.data = self.load_csv()
          elif self.file_extension == 'json':
               self.data = self.load_json()

     def load_csv(self) -> pd.DataFrame:
          try:
               data = pd.read_csv(self.file_path)
               return data
          except Exception as e:
               print(f"Error loading data: {e}")
               return pd.DataFrame()

     def load_json(self) -> pd.DataFrame:
          try:
               with open(self.file_path, 'r') as f:
                    data = json.load(f)
               return pd.DataFrame(data)
          except Exception as e:
               print(f"Error loading data: {e}")
               return pd.DataFrame()