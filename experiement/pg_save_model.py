from extend_path import sys
from _lib.database.postgres_conn import PostgresConn
import torch

pg_con = PostgresConn()

# Load model from local path
classifier_path = './artifact/model/intent_classifier_v3.pth'
extractor_path = './artifact/model/entity_extractor_v3.pth'

classifier = torch.load(classifier_path, weights_only=False, map_location=torch.device('cpu'))
extractor = torch.load(extractor_path, weights_only=False, map_location=torch.device('cpu'))

pg_con.save_model(classifier, model_name="intent_classifier_v3", model_type="classifier")
pg_con.save_model(extractor, model_name="extractor_v3", model_type="extractor")

pg_con.close()