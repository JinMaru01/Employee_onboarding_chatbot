import json
import os
from extend_path import sys
from _lib.database.postgres_conn import PostgresConn

# Define path to JSON file
json_path = "./artifact/data/json/response_design.json"

# Load knowledge base from file
try:
    with open(json_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except Exception as e:
    print(f"❌ Failed to load JSON from {json_path}: {e}")
    raise

# Initialize DB connection and save
pg_con = None
try:
    pg_con = PostgresConn()
    pg_con.save_knowledge(knowledge_base)
except Exception as db_err:
    print(f"❌ Failed to save knowledge base to PostgreSQL: {db_err}")
    raise
finally:
    if pg_con:
        pg_con.close()