import requests
import json
import datetime
import threading
import uuid

# Configuration from user request
FIREBASE_CONFIG = {
  "apiKey": "AIzaSyBRBbsKEyfuocJJhLAUKMRo4j8Kska5StM",
  "projectId": "xaidbms",
}

BASE_URL = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_CONFIG['projectId']}/databases/(default)/documents"

def _to_firestore_value(value):
    """Helper to convert python types to Firestore field types."""
    if isinstance(value, str):
        return {"stringValue": value}
    elif isinstance(value, bool):
        return {"booleanValue": value}
    elif isinstance(value, int):
        return {"integerValue": str(value)} # Firestore expects string for int64
    elif isinstance(value, float):
        return {"doubleValue": value}
    elif isinstance(value, dict):
        return {"mapValue": {"fields": {k: _to_firestore_value(v) for k, v in value.items()}}}
    elif isinstance(value, list):
        return {"arrayValue": {"values": [_to_firestore_value(v) for v in value]}}
    elif value is None:
        return {"nullValue": None}
    else:
        return {"stringValue": str(value)}

def log_to_firestore(collection: str, data: dict):
    """
    Logs data to a Firestore collection via REST API.
    Runs asynchronously to not block the main thread.
    """
    def _log():
        try:
            # add timestamp
            data['timestamp'] = datetime.datetime.utcnow().isoformat() + 'Z'
            
            # Convert to Firestore JSON format
            document = {
                "fields": {k: _to_firestore_value(v) for k, v in data.items()}
            }
            
            url = f"{BASE_URL}/{collection}?key={FIREBASE_CONFIG['apiKey']}"
            
            response = requests.post(url, json=document)
            
            if not response.ok:
                error_detail = response.text
                if response.status_code == 403:
                    print(f"⚠️ Firestore Permission Denied (403)")
                    print(f"   Please deploy firestore.rules to your Firebase project:")
                    print(f"   1. Install Firebase CLI: npm install -g firebase-tools")
                    print(f"   2. Login: firebase login")
                    print(f"   3. Initialize: firebase init firestore")
                    print(f"   4. Deploy rules: firebase deploy --only firestore:rules")
                    print(f"   Or update rules manually in Firebase Console")
                else:
                    print(f"⚠️ Firestore Log Warning: {response.status_code} - {error_detail}")
            # else:
            #     print(f"✓ Logged to {collection}")

        except Exception as e:
            print(f"⚠️ Firestore Log Error: {e}")

    thread = threading.Thread(target=_log)
    thread.start()

# --- Logging Helpers ---

def log_analysis_start(filename: str, target_column: str, analysis_id: str):
    log_to_firestore("user_activity", {
        "type": "analysis_start",
        "analysis_id": analysis_id,
        "filename": filename,
        "target_column": target_column
    })

def log_analysis_completion(analysis_id: str, model_name: str, metrics: dict, artifacts: dict):
    log_to_firestore("analyses", {
        "analysis_id": analysis_id,
        "model_name": model_name,
        "metrics": metrics,
        "artifacts": artifacts,
        "status": "completed"
    })

def log_llm_code_generation(prompt: str, code: str, success: bool, error: str = None):
    log_to_firestore("llm_calls", {
        "type": "code_generation",
        "prompt": prompt,
        "generated_code": code,
        "success": success,
        "error": error
    })

def log_query(query: str, answer: str, analysis_id: str):
    log_to_firestore("queries", {
        "query": query,
        "answer": answer,
        "analysis_id": analysis_id
    })
