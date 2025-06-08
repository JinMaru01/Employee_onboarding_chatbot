import os
from datetime import datetime, timedelta
from _lib.database.postgres_conn import get_connection

# Set up log file path
LOG_PATH = "logs"
LOG_DATE = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_PATH, f"{LOG_DATE}_chat_history.log")
os.makedirs(LOG_PATH, exist_ok=True)

def log_user_interaction(user_input, intent, confidence, bot_response):
    timestamp = datetime.now()

    # ---- File Logging ----
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(f"{timestamp} | User: {user_input}\n")
            log_file.write(f"{timestamp} | Predicted Intent: {intent} (confidence: {confidence:.4f})\n")
            log_file.write(f"{timestamp} | Bot Response: {bot_response}\n")
            log_file.write("-" * 80 + "\n")
    except Exception as file_err:
        print("Failed to write to log file:", file_err)

    # ---- PostgreSQL Logging ----
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SET search_path TO chatbot;")  # Optional, based on your schema
                cur.execute("""
                    INSERT INTO user_logs (timestamp, user_input, predicted_intent, confidence, bot_response)
                    VALUES (%s, %s, %s, %s, %s)
                """, (timestamp, user_input, intent, float(confidence), bot_response))
                conn.commit()
        except Exception as db_err:
            print("Failed to insert log into database:", db_err)
        finally:
            conn.close()

def load_chat_history(date_str=None, limit=50):
    """
    Fetch latest chat history from the database for a given date (YYYY-MM-DD).
    If no date is provided, fetch latest N logs regardless of date.
    """
    conn = get_connection()
    history = []

    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SET search_path TO chatbot;")

                if date_str:
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        start_ts = date_obj
                        end_ts = date_obj + timedelta(days=1)
                    except ValueError:
                        raise ValueError("Invalid date format. Use YYYY-MM-DD.")

                    cur.execute("""
                        SELECT timestamp, user_input, predicted_intent, confidence, bot_response
                        FROM user_logs
                        WHERE timestamp >= %s AND timestamp < %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (start_ts, end_ts, limit))
                else:
                    cur.execute("""
                        SELECT timestamp, user_input, predicted_intent, confidence, bot_response
                        FROM user_logs
                        ORDER BY timestamp DESC
                        LIMIT %s
                    """, (limit,))

                rows = cur.fetchall()
                for row in rows:
                    history.append({
                        "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                        "user_input": row["user_input"],
                        "predicted_intent": row["predicted_intent"],
                        "confidence": float(row["confidence"]),
                        "bot_response": row["bot_response"]
                    })

        except Exception as e:
            import traceback
            print("Failed to load chat history:", str(e))
            traceback.print_exc()
            raise
        finally:
            conn.close()

    return list(reversed(history))

def get_unique_dates():
    """
    Returns a list of unique dates (YYYY-MM-DD) from the user_logs table.
    """
    conn = get_connection()
    unique_dates = []

    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SET search_path TO chatbot;")
                cur.execute("""
                    SELECT DISTINCT DATE(timestamp) AS date
                    FROM user_logs
                    ORDER BY date DESC;
                """)
                rows = cur.fetchall()
                unique_dates = [row["date"].isoformat() for row in rows]
        except Exception as e:
            import traceback
            print("Failed to fetch unique dates:", str(e))
            traceback.print_exc()
            raise
        finally:
            conn.close()

    return unique_dates

def delete_chat_history_by_date(date_str):
    """
    Deletes chat logs from both PostgreSQL and log file for the specified date (YYYY-MM-DD).
    """
    try:
        # Validate date
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        start_ts = date_obj
        end_ts = date_obj + timedelta(days=1)
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    # ---- Delete from PostgreSQL ----
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SET search_path TO chatbot;")
                cur.execute("""
                    DELETE FROM user_logs
                    WHERE timestamp >= %s AND timestamp < %s
                """, (start_ts, end_ts))
                conn.commit()
                print(f"✅ Deleted logs from database for {date_str}")
        except Exception as db_err:
            print(f"❌ Failed to delete logs from database for {date_str}:", db_err)
        finally:
            conn.close()

    # ---- Delete from log file ----
    log_file_path = os.path.join(LOG_PATH, f"{date_str}_chat_history.log")
    try:
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
            print(f"✅ Deleted local log file: {log_file_path}")
        else:
            print(f"ℹ️ No log file found for {date_str}")
    except Exception as file_err:
        print(f"❌ Failed to delete log file for {date_str}:", file_err)
