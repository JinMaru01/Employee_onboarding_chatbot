import json
from extend_path import sys
from _lib.database.postgres_conn import PostgresConn

# Example: load your entire list from a file or variable
with open("./artifact/data/json/response_design.json", "r") as f:
    knowledge_base = json.load(f)

pg_con = PostgresConn()

try:
    pg_con.save_knowledge(knowledge_base)
finally:
    pg_con.close()