import os
import io
import torch
import joblib
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError

# Load .env variables
load_dotenv()

class MinioConn:
    def __init__(self):
        self.bucket = os.getenv("MINIO_BUCKET", "models")
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin123"),
            secure=False
        )

    def bucket_exists(self, bucket_name):
        """Checks if a bucket exists."""
        try:
            return self.client.bucket_exists(bucket_name)
        except S3Error as e:
            print(f"❌ Error checking bucket existence: {e}")
            return False

    def upload_model(self, model, bucket_name, object_name):
        """Uploads a PyTorch model's state_dict to the given bucket."""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        self.client.put_object(
            bucket_name,
            object_name,
            data=buffer,
            length=buffer.getbuffer().nbytes,
            content_type="application/octet-stream"
        )
        print(f"✅ Model uploaded to bucket '{bucket_name}' as '{object_name}'")

    def load_model(self, model_class, bucket_name, object_name, model_ckpt=None, num_labels=2, map_location="cpu"):
        """Loads a PyTorch model from MinIO directly into memory."""
        try:
            response = self.client.get_object(bucket_name, object_name)
            model_bytes = response.read()
            buffer = io.BytesIO(model_bytes)

            if model_ckpt:
                model = model_class.from_pretrained(model_ckpt, num_labels=num_labels)
                model.load_state_dict(torch.load(buffer, map_location=map_location))
            else:
                model = torch.load(buffer, map_location=map_location)

            model.eval()
            print(f"✅ Model loaded from bucket '{bucket_name}' as '{object_name}'")
            return model
        except S3Error as e:
            print(f"❌ Failed to load model: {e}")
            return None