"""
Simple direct test of query handler with verbose output
"""
import os
import sys
import pickle
import json
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from explainable_dbms.query_handler import analyze_and_answer_query

# Load environment
load_dotenv()

# Use specific analysis
analysis_id = "d3be70e7-1b35-4b15-a7a9-c67323724a64"
analysis_path = Path("artifacts") / analysis_id

print(f"\n{'='*80}")
print(f"Testing Query Handler with Analysis: {analysis_id}")
print(f"{'='*80}\n")

# Load state
with open(analysis_path / "state.pkl", "rb") as f:
    state = pickle.load(f)

print(f"✓ State loaded")
print(f"  Target: {state['target_column']}")
print(f"  Task: {state['task_type']}")
print(f"  Data shape: {state['user_df'].shape}\n")

# Load metrics
with open(analysis_path / "metrics.json", "r") as f:
    metrics = json.load(f)
print(f"✓ Metrics: {metrics}\n")

# List artifacts
artifacts = {}
for file_path in analysis_path.glob('*'):
    if file_path.suffix in ['.png', '.json']:
        artifacts[file_path.name] = str(file_path)
print(f"✓ Artifacts: {list(artifacts.keys())}\n")

# Test query
query = "What is the model's performance and which features are most important?"

print(f"{'='*80}")
print(f"QUERY: {query}")
print(f"{'='*80}\n")

answer = analyze_and_answer_query(
    query,
    state["user_df"],
    metrics,
    artifacts,
    state["target_column"],
    state["task_type"]
)

print(f"\n{'='*80}")
print(f"FINAL ANSWER:")
print(f"{'='*80}")
print(answer)
print(f"{'='*80}\n")
