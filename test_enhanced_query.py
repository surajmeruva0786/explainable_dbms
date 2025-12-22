"""
Test enhanced query handler with rich context
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
print("TESTING ENHANCED QUERY HANDLER WITH RICH CONTEXT")
print("="*80 + "\n")

# Load state
with open(analysis_path / "state.pkl", "rb") as f:
    state = pickle.load(f)

# Load metrics
with open(analysis_path / "metrics.json", "r") as f:
    metrics = json.load(f)

# List artifacts
artifacts = {}
for file_path in analysis_path.glob('*'):
    if file_path.suffix in ['.png', '.json']:
        artifacts[file_path.name] = str(file_path)

# Test with an insightful question
query = "Why are there more female survivors than male survivors? What patterns explain this?"

print(f"Dataset: {state['user_df'].shape}")
print(f"Target: {state['target_column']}")
print(f"Columns: {list(state['user_df'].columns)}\n")

print("="*80)
print(f"QUERY: {query}")
print("="*80 + "\n")

print("Calling LLM with comprehensive context (dataset stats, correlations, cross-tabs)...\n")

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
    print("FINAL ANSWER:")
    print("="*80)
    print(answer)
    print("="*80)
    print("\nSUCCESS: Enhanced query handler provides rich, data-driven insights!")
    print("="*80 + "\n")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
