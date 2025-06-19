import json
import psycopg2
from psycopg2.extras import RealDictCursor
import io
import os
import torch
import joblib
from transformers import DistilBertForSequenceClassification, DistilBertForTokenClassification

from dotenv import load_dotenv

load_dotenv()

class PostgresConn:
    def __init__(self, search_path=os.getenv("PG_SCHEMA")):
        try:
            self.conn = psycopg2.connect(
                dbname=os.getenv("POSTGRES_DB"),
                user=os.getenv("POSTGRES_USER"),
                password=os.getenv("POSTGRES_PASSWORD"),
                host=os.getenv("SERVER_HOST"),
                port=os.getenv("POSTGRES_PORT"),
                cursor_factory=RealDictCursor
            )
            print("✅ Postgres connection established successfully!")

            if search_path:
                with self.conn.cursor() as cur:
                    cur.execute(f"SET search_path TO {search_path}")
                print(f"✅ Search path set to: {search_path}")
        except Exception as e:
            print("❌ Failed to connect to Postgres:", e)
            self.conn = None
            raise

    def get_connection(self):
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
            print("🔒 Postgres connection closed.")

    def save_knowledge(self, knowledge_list):
        """
        Save or update a list of knowledge objects in the chatbot_knowledge table.
        Each item in knowledge_list must have 'intent', 'entities', and 'responses'.
        """
        if self.conn is None:
            raise ConnectionError("No database connection.")

        insert_query = """
            INSERT INTO chatbot_knowledge (intent, entities, responses)
            VALUES (%s, %s, %s)
            ON CONFLICT (intent) DO UPDATE
            SET entities = EXCLUDED.entities,
                responses = EXCLUDED.responses,
                updated_at = now()
        """

        try:
            with self.conn:
                with self.conn.cursor() as cur:
                    for data in knowledge_list:
                        cur.execute(insert_query, (
                            data["intent"],
                            json.dumps(data["entities"]),
                            json.dumps(data["responses"])
                        ))
            print(f"✅ Saved {len(knowledge_list)} knowledge entries successfully.")
        except Exception as e:
            print("❌ Failed to save knowledge base:", e)
            raise

    def load_knowledge_base(self):
        """
        Load all knowledge base entries from the chatbot_knowledge table.
        Returns a list of knowledge objects (with parsed JSON for entities and responses).
        """
        query = "SELECT intent, entities, responses FROM chatbot_knowledge"

        try:
            with self.conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()

            knowledge_base = []
            for row in rows:
                knowledge_base.append({
                    "intent": row["intent"],
                    "entities": (row["entities"]) if row["entities"] else {},
                    "responses": (row["responses"]) if row["responses"] else {}
                })
            print(f"✅ Loaded {len(knowledge_base)} knowledge entries from PostgreSQL.")
            return knowledge_base

        except Exception as e:
            print("❌ Failed to load knowledge base:", e)
            raise

    def save_model(self, model, model_name, model_type):
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        binary_data = buffer.getvalue()

        query = """
            INSERT INTO model_store (model_name, model_type, model_data)
            VALUES (%s, %s, %s)
            ON CONFLICT (model_name) DO UPDATE
            SET model_type = EXCLUDED.model_type,
                model_data = EXCLUDED.model_data,
                updated_at = now()
        """
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute(query, (model_name, model_type, binary_data))
        print(f"✅ Model '{model_name}' saved to PostgreSQL.")

    def load_classifier(self, model_name, num_labels=12, model_ckpt="distilbert-base-uncased"):
        query = "SELECT model_data FROM model_store WHERE model_name = %s AND model_type = 'classifier'"
        with self.conn.cursor() as cur:
            cur.execute(query, (model_name,))
            result = cur.fetchone()

        if not result:
            raise ValueError(f"❌ Classifier '{model_name}' not found in database!")

        model = DistilBertForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
        buffer = io.BytesIO(result["model_data"])
        model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
        model.eval()
        print(f"✅ Classifier model '{model_name}' loaded from PostgreSQL.")
        return model

    def load_extractor(self, model_name, num_labels=26, model_ckpt="distilbert-base-uncased"):
        query = "SELECT model_data FROM model_store WHERE model_name = %s AND model_type = 'extractor'"
        with self.conn.cursor() as cur:
            cur.execute(query, (model_name,))
            result = cur.fetchone()

        if not result:
            raise ValueError(f"❌ Extractor '{model_name}' not found in database!")

        model = DistilBertForTokenClassification.from_pretrained(model_ckpt, num_labels=num_labels)
        buffer = io.BytesIO(result["model_data"])
        model.load_state_dict(torch.load(buffer, map_location=torch.device("cpu")))
        model.eval()
        print(f"✅ Extractor model '{model_name}' loaded from PostgreSQL.")
        return model

    def save_artifact(self, name, artifact_obj, artifact_type="generic"):
        """
        Save any picklable Python object (e.g. tokenizer, encoder, tensor).
        """
        buffer = io.BytesIO()
        joblib.dump(artifact_obj, buffer)
        binary_data = buffer.getvalue()

        query = """
            INSERT INTO artifact_store (name, artifact_type, artifact_data)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO UPDATE
            SET artifact_type = EXCLUDED.artifact_type,
                artifact_data = EXCLUDED.artifact_data,
                updated_at = NOW()
        """
        with self.conn:
            with self.conn.cursor() as cur:
                cur.execute(query, (name, artifact_type, binary_data))
        print(f"✅ Saved artifact '{name}' ({artifact_type}) to PostgreSQL.")

    def load_artifact(self, name):
        """
        Load and return a Python object stored in artifact_store by name.
        """
        query = "SELECT artifact_data FROM artifact_store WHERE name = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (name,))
            result = cur.fetchone()

        if not result:
            raise ValueError(f"❌ Artifact '{name}' not found in PostgreSQL!")

        buffer = io.BytesIO(result["artifact_data"])
        obj = joblib.load(buffer)
        print(f"✅ Loaded artifact '{name}' from PostgreSQL.")
        return obj