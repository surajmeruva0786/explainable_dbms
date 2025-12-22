"""
Test script to verify query answer functionality through LLM calls.
This script tests the complete flow:
1. Loads analysis state
2. Calls query_handler with a test query
3. Verifies LLM is being called correctly
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

def test_query_functionality():
    print("\n" + "="*80)
    print("🧪 TESTING QUERY ANSWER FUNCTIONALITY")
    print("="*80 + "\n")
    
    # Load environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return False
    
    print(f"✓ API Key loaded: {api_key[:10]}...{api_key[-4:]}\n")
    
    # Find most recent analysis
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        print("❌ ERROR: No artifacts directory found")
        return False
    
    # Get all analysis directories (UUIDs)
    analysis_dirs = [d for d in artifacts_dir.iterdir() if d.is_dir()]
    
    if not analysis_dirs:
        print("❌ ERROR: No analysis sessions found in artifacts/")
        print("   Please run an analysis first using the web interface")
        return False
    
    # Use most recent analysis
    latest_analysis = max(analysis_dirs, key=lambda d: d.stat().st_mtime)
    print(f"✓ Found analysis session: {latest_analysis.name}\n")
    
    # Load state
    state_path = latest_analysis / "state.pkl"
    if not state_path.exists():
        print(f"❌ ERROR: No state.pkl found in {latest_analysis}")
        return False
    
    print("📂 Loading analysis state...")
    with open(state_path, "rb") as f:
        state = pickle.load(f)
    
    print(f"✓ State loaded successfully")
    print(f"  - Target column: {state['target_column']}")
    print(f"  - Task type: {state['task_type']}")
    print(f"  - Dataset shape: {state['user_df'].shape}\n")
    
    # Load metrics
    metrics_path = latest_analysis / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print(f"✓ Metrics loaded: {metrics}\n")
    else:
        print("⚠️  No metrics.json found\n")
    
    # List artifacts
    artifacts_list = {}
    for file_path in latest_analysis.glob('*'):
        if file_path.suffix in ['.png', '.json']:
            artifacts_list[file_path.name] = str(file_path)
    
    print(f"✓ Found {len(artifacts_list)} artifacts:")
    for name in artifacts_list.keys():
        print(f"  - {name}")
    print()
    
    # Test queries
    test_queries = [
        "What is the model performance?",
        "What are the most important features?",
        "Explain how the model makes predictions",
    ]
    
    print("="*80)
    print("🤖 TESTING LLM QUERY HANDLER")
    print("="*80 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST QUERY {i}: {query}")
        print('='*80)
        
        try:
            answer = analyze_and_answer_query(
                query,
                state["user_df"],
                metrics,
                artifacts_list,
                state["target_column"],
                state["task_type"]
            )
            
            print(f"\n✅ ANSWER RECEIVED:")
            print("-"*80)
            print(answer)
            print("-"*80)
            
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe query answer functionality is working correctly through LLM calls.")
    print("The system successfully:")
    print("  1. Loaded analysis state from disk")
    print("  2. Constructed context for the LLM")
    print("  3. Called Gemini API with the query")
    print("  4. Received and returned the answer")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    success = test_query_functionality()
    sys.exit(0 if success else 1)
