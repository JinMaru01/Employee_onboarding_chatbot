import os
import io
import torch
import joblib
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Load .env variables
load_dotenv()

class MinioConn:
    def __init__(self):
        self.bucket = os.getenv("MINIO_BUCKET", "models")
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        )

        # Ensure bucket exists
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket)
            print(f"✅ Bucket '{self.bucket}' created!")

    def label_encoder_save(self, label_encoder, file_name):
        buffer = io.BytesIO()
        joblib.dump(label_encoder, buffer)
        buffer.seek(0)
        self.s3_client.upload_fileobj(buffer, self.bucket, file_name)
        print(f"✅ LabelEncoder saved to MinIO as '{file_name}'")

    def label_encoder_load(self, file_name):
        buffer = io.BytesIO()
        try:
            self.s3_client.download_fileobj(self.bucket, file_name, buffer)
            buffer.seek(0)
            label_encoder = joblib.load(buffer)
            return label_encoder
        except ClientError:
            raise ValueError("❌ LabelEncoder not found in MinIO!")

    def model_save(self, model, file_name):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self.s3_client.upload_fileobj(buffer, self.bucket, file_name)
        print(f"✅ Model state_dict saved to MinIO as '{file_name}'")

    def classifier_load(self, file_name, num_labels=12, model_ckpt="distilbert-base-uncased"):
        from transformers import DistilBertForSequenceClassification
        buffer = io.BytesIO()
        try:
            self.s3_client.download_fileobj(self.bucket, file_name, buffer)
            buffer.seek(0)
            model = DistilBertForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
            model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
            model.eval()
            print(f"✅ Classifier loaded from MinIO file '{file_name}'")
            return model
        except ClientError:
            raise ValueError("❌ Classifier model not found in MinIO!")

    def extractor_load(self, file_name, num_labels=26, model_ckpt="distilbert-base-uncased"):
        from transformers import DistilBertForTokenClassification
        buffer = io.BytesIO()
        try:
            self.s3_client.download_fileobj(self.bucket, file_name, buffer)
            buffer.seek(0)
            model = DistilBertForTokenClassification.from_pretrained(model_ckpt, num_labels=num_labels)
            model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
            model.eval()
            print(f"✅ Extractor loaded from MinIO file '{file_name}'")
            return model
        except ClientError:
            raise ValueError("❌ Extractor model not found in MinIO!")
