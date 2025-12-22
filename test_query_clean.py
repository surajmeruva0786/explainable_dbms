"""
Clean test output for query functionality
"""
import os
import sys
import pickle
import json
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))
from explainable_dbms.query_handler import analyze_and_answer_query

load_dotenv()

# Use specific analysis
analysis_id = "d3be70e7-1b35-4b15-a7a9-c67323724a64"
analysis_path = Path("artifacts") / analysis_id

print("\n" + "="*80)
print("QUERY ANSWER FUNCTIONALITY TEST")
print("="*80 + "\n")

# Load state
with open(analysis_path / "state.pkl", "rb") as f:
    state = pickle.load(f)

print(f"Analysis ID: {analysis_id}")
print(f"Target Column: {state['target_column']}")
print(f"Task Type: {state['task_type']}")
print(f"Dataset Shape: {state['user_df'].shape}")

# Load metrics
with open(analysis_path / "metrics.json", "r") as f:
    metrics = json.load(f)
print(f"Metrics: {json.dumps(metrics, indent=2)}")

# List artifacts
artifacts = {}
for file_path in analysis_path.glob('*'):
    if file_path.suffix in ['.png', '.json']:
        artifacts[file_path.name] = str(file_path)
print(f"\nArtifacts found: {len(artifacts)}")

# Test query
query = "What is the model performance?"

print("\n" + "="*80)
print(f"QUERY: {query}")
print("="*80 + "\n")

print("Calling LLM (Gemini 2.5 Flash)...\n")

try:
    answer = analyze_and_answer_query(
        query,
        state["user_df"],
        metrics,
        artifacts,
        state["target_column"],
        state["task_type"]
    )
    
    print("\n" + "="*80)
    print("ANSWER RECEIVED:")
    print("="*80)
    print(answer)
    print("="*80)
    print("\nSUCCESS: Query answer functionality is working through LLM calls!")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
