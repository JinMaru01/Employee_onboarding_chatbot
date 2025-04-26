import io
import uuid
import torch
import joblib
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from transformers import DistilBertForSequenceClassification

class QdrantModelStore:
    def __init__(self, host='localhost', port=6333, collection_name='models'):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        if self.collection not in [col.name for col in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE)  # dummy vector
            )

    def save_model(self, model: torch.nn.Module, model_type: str):
        """
        Save a model's state_dict into Qdrant as bytes under a unique UUID
        """
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model_blob = buffer.read()

        # Save with a dummy vector (required by Qdrant)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],  # dummy vector
            payload={
                "model_type": model_type,
                "model_blob": model_blob
            }
        )

        self.client.upsert(
            collection_name=self.collection,
            points=[point]
        )
        print(f"✅ Model '{model_type}' saved to Qdrant.")

    def fetch_model_state_dict(self, model_type: str):
        hits = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="model_type", match=MatchValue(value=model_type))]
            ),
            limit=1
        )

        if not hits or not hits[0]:
            raise ValueError(f"❌ Model with type '{model_type}' not found in Qdrant.")

        model_blob = hits[0][0].payload.get("model_blob")
        if not model_blob:
            raise ValueError(f"❌ No 'model_blob' found in payload for model_type='{model_type}'")

        print(f"✅ Model blob for '{model_type}' retrieved from Qdrant.")
        return model_blob

    def load_model(self, model_type: str, num_labels: int = 10):
        model_blob = self.fetch_model_state_dict(model_type)
        buffer = io.BytesIO(model_blob)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=num_labels
        )
        model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
        model.eval()

        print(f"✅ DistilBERT model '{model_type}' loaded from Qdrant.")
        return model
