import io
import _lib.database.redis_conn as redis_conn
import torch
import joblib
import numpy as np
import json
import pickle
from typing import Dict, List, Union, Tuple, Optional
from transformers import DistilBertForSequenceClassification

class RedisConn:
    def __init__(self, host='localhost', port=6379, db=0, password=None, prefix='nlp:'):
        """
        Initialize Redis connection with configurable parameters
        
        Args:
            host: Redis host address
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
            prefix: Prefix for all keys stored by this instance
        """
        self.redis_client = redis_conn.Redis(
            host=host, 
            port=port, 
            db=db,
            password=password,
            decode_responses=False  # Keep binary data as is
        )
        self.prefix = prefix
        
        # Test connection
        try:
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {host}:{port}")
        except redis_conn.ConnectionError as e:
            print(f"❌ Failed to connect to Redis: {e}")
    
    def _get_key(self, key: str) -> str:
        """Prepend the instance prefix to the key"""
        return f"{self.prefix}{key}"
    
    def model_save(self, model, file_name: str) -> bool:
        """
        Save model state_dict to Redis
        
        Args:
            model: PyTorch model object
            file_name: Key name to store the model under
            
        Returns:
            bool: Success status
        """
        try:
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            self.redis_client.set(self._get_key(file_name), buffer.getvalue())
            print(f"✅ Model state saved in Redis with key: {self._get_key(file_name)}")
            return True
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False

    def model_load(self, model_name: str, num_labels: int = 20) -> Optional[DistilBertForSequenceClassification]:
        """
        Load model state_dict from Redis without saving locally
        
        Args:
            model_name: Key name of the stored model
            num_labels: Number of classification labels
            
        Returns:
            DistilBertForSequenceClassification: Loaded model
        """
        try:
            model_bytes = self.redis_client.get(self._get_key(model_name))
            if model_bytes is None:
                raise ValueError(f"❌ Model not found in Redis with key: {self._get_key(model_name)}")
            
            buffer = io.BytesIO(model_bytes)
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
            model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))
            model.eval()
            print("✅ Model successfully loaded from Redis and ready for inference!")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return None

    def save_label_encoder(self, label_encoder, file_name: str) -> bool:
        """
        Save the fitted LabelEncoder to Redis
        
        Args:
            label_encoder: Fitted sklearn LabelEncoder
            file_name: Key name to store the encoder under
            
        Returns:
            bool: Success status
        """
        try:
            buffer = io.BytesIO()
            joblib.dump(label_encoder, buffer)
            self.redis_client.set(self._get_key(file_name), buffer.getvalue())
            print(f"✅ LabelEncoder saved in Redis with key: {self._get_key(file_name)}")
            
            # Also save class names as JSON for easier access
            classes = label_encoder.classes_.tolist()
            self.redis_client.set(
                self._get_key(f"{file_name}_classes"), 
                json.dumps(classes).encode('utf-8')
            )
            return True
        except Exception as e:
            print(f"❌ Error saving label encoder: {str(e)}")
            return False

    def load_label_encoder(self, key: str):
        """
        Load the fitted LabelEncoder from Redis
        
        Args:
            key: Key name of the stored encoder
            
        Returns:
            LabelEncoder: Loaded encoder
        """
        try:
            encoder_bytes = self.redis_client.get(self._get_key(key))
            if encoder_bytes is None:
                raise ValueError(f"❌ LabelEncoder not found in Redis with key: {self._get_key(key)}")
            
            buffer = io.BytesIO(encoder_bytes)
            label_encoder = joblib.load(buffer)
            print(f"✅ LabelEncoder successfully loaded from Redis with classes: {label_encoder.classes_}")
            return label_encoder
        except Exception as e:
            print(f"❌ Error loading label encoder: {str(e)}")
            return None
            
    def save_embedding(self, embedding: np.ndarray, label: int, idx: Union[int, str], 
                      metadata: Optional[Dict] = None) -> bool:
        """
        Save a single embedding vector to Redis
        
        Args:
            embedding: Numpy array of the embedding vector
            label: Integer label associated with this embedding
            idx: Unique identifier for this embedding
            metadata: Optional dict of additional metadata to store
            
        Returns:
            bool: Success status
        """
        try:
            key = f"embedding:{idx}"
            data = {
                'embedding': embedding.tobytes(),
                'shape': embedding.shape,
                'dtype': str(embedding.dtype),
                'label': int(label)
            }
            
            # Add optional metadata if provided
            if metadata:
                data['metadata'] = metadata
                
            serialized_data = pickle.dumps(data)
            self.redis_client.set(self._get_key(key), serialized_data)
            return True
        except Exception as e:
            print(f"❌ Error saving embedding: {str(e)}")
            return False
    
    def save_embeddings_batch(self, embeddings: np.ndarray, labels: List[int], 
                             start_idx: int = 0, metadata_list: Optional[List[Dict]] = None) -> bool:
        """
        Save a batch of embeddings to Redis
        
        Args:
            embeddings: Numpy array of shape (n_samples, embedding_dim)
            labels: List of integer labels for each embedding
            start_idx: Starting index for embedding keys
            metadata_list: Optional list of metadata dicts for each embedding
            
        Returns:
            bool: Success status
        """
        try:
            for i, embedding in enumerate(embeddings):
                idx = start_idx + i
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                self.save_embedding(embedding, labels[i], idx, metadata)
                
            print(f"✅ Batch of {len(embeddings)} embeddings saved to Redis")
            return True
        except Exception as e:
            print(f"❌ Error saving embedding batch: {str(e)}")
            return False
    
    def load_embedding(self, idx: Union[int, str]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[Dict]]:
        """
        Load a single embedding vector from Redis
        
        Args:
            idx: Unique identifier for the embedding
            
        Returns:
            Tuple[np.ndarray, int, dict]: Embedding vector, label, and metadata
        """
        try:
            key = f"embedding:{idx}"
            serialized_data = self.redis_client.get(self._get_key(key))
            
            if serialized_data is None:
                print(f"⚠️ No embedding found with key: {self._get_key(key)}")
                return None, None, None
                
            data = pickle.loads(serialized_data)
            embedding = np.frombuffer(data['embedding'], dtype=np.dtype(data['dtype'])).reshape(data['shape'])
            label = data['label']
            metadata = data.get('metadata', {})
            
            return embedding, label, metadata
        except Exception as e:
            print(f"❌ Error loading embedding: {str(e)}")
            return None, None, None
    
    def load_embeddings_batch(self, indices: List[Union[int, str]]) -> List[Tuple[np.ndarray, int, Dict]]:
        """
        Load multiple embeddings from Redis
        
        Args:
            indices: List of indices to load
            
        Returns:
            List of (embedding, label, metadata) tuples
        """
        results = []
        for idx in indices:
            embedding, label, metadata = self.load_embedding(idx)
            if embedding is not None:
                results.append((embedding, label, metadata))
        
        print(f"✅ Loaded {len(results)} embeddings from Redis")
        return results
    
    def save_metadata(self, key: str, metadata: Dict) -> bool:
        """
        Save metadata dictionary to Redis
        
        Args:
            key: Key name for the metadata
            metadata: Dictionary of metadata to store
            
        Returns:
            bool: Success status
        """
        try:
            self.redis_client.set(
                self._get_key(key), 
                json.dumps(metadata).encode('utf-8')
            )
            print(f"✅ Metadata saved to Redis with key: {self._get_key(key)}")
            return True
        except Exception as e:
            print(f"❌ Error saving metadata: {str(e)}")
            return False
    
    def load_metadata(self, key: str) -> Optional[Dict]:
        """
        Load metadata dictionary from Redis
        
        Args:
            key: Key name of the stored metadata
            
        Returns:
            dict: Loaded metadata dictionary
        """
        try:
            data = self.redis_client.get(self._get_key(key))
            if data is None:
                print(f"⚠️ No metadata found with key: {self._get_key(key)}")
                return None
                
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            print(f"❌ Error loading metadata: {str(e)}")
            return None
    
    def get_embedding_keys(self, pattern: str = "embedding:*") -> List[str]:
        """
        Get all keys matching a pattern
        
        Args:
            pattern: Pattern to match keys against
            
        Returns:
            List[str]: List of matching keys
        """
        try:
            full_pattern = self._get_key(pattern)
            keys = self.redis_client.keys(full_pattern)
            # Strip prefix from keys for easier use
            prefix_len = len(self.prefix)
            return [key.decode('utf-8')[prefix_len:] for key in keys]
        except Exception as e:
            print(f"❌ Error getting keys: {str(e)}")
            return []
    
    def delete_key(self, key: str) -> bool:
        """
        Delete a key from Redis
        
        Args:
            key: Key to delete
            
        Returns:
            bool: Success status
        """
        try:
            result = self.redis_client.delete(self._get_key(key))
            if result > 0:
                print(f"✅ Key {self._get_key(key)} deleted from Redis")
                return True
            else:
                print(f"⚠️ Key {self._get_key(key)} not found in Redis")
                return False
        except Exception as e:
            print(f"❌ Error deleting key: {str(e)}")
            return False
    
    def flush_keys(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern
        
        Args:
            pattern: Pattern to match keys against
            
        Returns:
            int: Number of keys deleted
        """
        try:
            full_pattern = self._get_key(pattern)
            keys = self.redis_client.keys(full_pattern)
            
            if not keys:
                print(f"⚠️ No keys found matching pattern: {full_pattern}")
                return 0
                
            deleted = 0
            for key in keys:
                self.redis_client.delete(key)
                deleted += 1
                
            print(f"✅ Deleted {deleted} keys matching pattern: {full_pattern}")
            return deleted
        except Exception as e:
            print(f"❌ Error flushing keys: {str(e)}")
            return 0
    
    def get_nearest_embeddings(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """
        Find nearest embeddings to a query embedding using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of nearest embeddings to return
            
        Returns:
            List of (embedding_id, similarity_score, label) tuples
        """
        try:
            # Get all embedding keys
            all_keys = self.get_embedding_keys("embedding:*")
            
            if not all_keys:
                print("⚠️ No embeddings found in Redis")
                return []
            
            results = []
            for key in all_keys:
                # Extract index from key
                idx = key.split(":")[-1]
                
                # Load embedding
                embedding, label, _ = self.load_embedding(idx)
                if embedding is None:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )
                
                results.append((idx, float(similarity), label))
            
            # Sort by similarity (highest first) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
        except Exception as e:
            print(f"❌ Error finding nearest embeddings: {str(e)}")
            return []

    def save_model_complete(self, model, model_key: str, label_encoder, encoder_key: str, 
                          metadata: Optional[Dict] = None) -> bool:
        """
        Save complete model, label encoder, and metadata in one operation
        
        Args:
            model: PyTorch model
            model_key: Key to store model under
            label_encoder: Label encoder
            encoder_key: Key to store encoder under
            metadata: Optional metadata to store
            
        Returns:
            bool: Success status
        """
        try:
            # Save model
            model_result = self.model_save(model, model_key)
            
            # Save label encoder
            encoder_result = self.save_label_encoder(label_encoder, encoder_key)
            
            # Save metadata if provided
            metadata_result = True
            if metadata:
                metadata_result = self.save_metadata(f"{model_key}_metadata", metadata)
                
            return model_result and encoder_result and metadata_result
        except Exception as e:
            print(f"❌ Error in complete model save: {str(e)}")
            return False
            
    def load_model_complete(self, model_key: str, encoder_key: str, 
                          num_labels: Optional[int] = None) -> Tuple[Optional[DistilBertForSequenceClassification], Optional, Optional[Dict]]:
        """
        Load complete model, label encoder, and metadata in one operation
        
        Args:
            model_key: Key where model is stored
            encoder_key: Key where encoder is stored
            num_labels: Number of labels (if None, will try to infer from metadata)
            
        Returns:
            Tuple of (model, label_encoder, metadata)
        """
        try:
            # Try to load metadata first to get num_labels if not provided
            metadata = self.load_metadata(f"{model_key}_metadata")
            
            if num_labels is None:
                if metadata and 'num_labels' in metadata:
                    num_labels = metadata['num_labels']
                else:
                    # Default value if we can't determine
                    num_labels = 20
                    print(f"⚠️ Using default num_labels={num_labels}")
            
            # Load model
            model = self.model_load(model_key, num_labels=num_labels)
            
            # Load label encoder
            label_encoder = self.load_label_encoder(encoder_key)
            
            return model, label_encoder, metadata
        except Exception as e:
            print(f"❌ Error in complete model load: {str(e)}")
            return None, None, None