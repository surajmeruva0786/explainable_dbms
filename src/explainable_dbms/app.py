import uvicorn
import webbrowser
import threading
import pickle
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
import json

from .part1_setup import initialize_database
from .part2_schema_data import load_user_data, compute_aggregated_features
from .part3_ml_xai import train_models_and_explain
from .part4_visualization import generate_visualizations
from .query_handler import analyze_and_answer_query
from .llm_model_selector import select_model_with_llm

from .llm_column_extractor import extract_target_columns_with_llm

# --- Directory setup for state management ---
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("temp_data")
DATA_DIR.mkdir(exist_ok=True)

# --- FastAPI App and Data Models ---
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Mount artifacts directory for serving generated plots
app.mount("/artifacts", StaticFiles(directory="artifacts"), name="artifacts")

class AnalysisRequest(BaseModel):
    filename: str
    target_column: str

class QueryRequest(BaseModel):
    query: str
    analysis_id: str

# --- Core Logic (adapted for API) ---

def run_analysis(filename: str, target_column: str):
    """
    This function contains the core ML pipeline. It's adapted to save
    its results to disk instead of returning them to a stateful UI.
    """
    # 1. Setup & Data Loading
    engine = initialize_database()
    dataset_path = DATA_DIR / filename
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {filename}")
    user_df = load_user_data(engine, str(dataset_path), None)
    table_name = dataset_path.stem

    # 3. Determine task type
    target_series = user_df[target_column]
    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
        task_type = "regression"
    else:
        task_type = "classification"
    
    # 4. Feature Engineering
    feature_df = compute_aggregated_features(engine, table_name, None)

    # 5. LLM Model Selection
    df_head = user_df.head().to_string()
    selected_model = select_model_with_llm(df_head, target_column, task_type)

    # 6. Model Training
    artifact = train_models_and_explain(engine, feature_df, target_column, task_type, selected_model, None)

    # 7. Visualization
    visualizations = generate_visualizations([artifact])
    
    # 8. Save state for later queries
    analysis_id = str(uuid.uuid4())
    analysis_path = ARTIFACTS_DIR / analysis_id
    analysis_path.mkdir()

    # Save plots and get their web-accessible paths
    plot_urls = {}
    for name, plot in visualizations[artifact.model_name].items():
        plot_path = analysis_path / f"{name}.png"
        plot.savefig(plot_path)
        plot_urls[name] = f"/artifacts/{analysis_id}/{name}.png" # URL for frontend

    # Save other necessary data using pickle
    state_to_save = {
        "artifact": artifact,
        "user_df": user_df,
        "feature_df": feature_df,
        "target_column": target_column,
        "task_type": task_type
    }
    with open(analysis_path / "state.pkl", "wb") as f:
        pickle.dump(state_to_save, f)

    return {
        "message": f"Analysis complete. Selected model: {artifact.model_name}.",
        "analysis_id": analysis_id,
        "plots": plot_urls
    }

def handle_query(query: str, analysis_id: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # Load the state for this analysis
    state_path = ARTIFACTS_DIR / analysis_id / "state.pkl"
    if not state_path.exists():
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    with open(state_path, "rb") as f:
        state = pickle.load(f)

    # Load metrics
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    # List available artifacts
    artifacts_list = {}
    analysis_dir = ARTIFACTS_DIR / analysis_id
    if analysis_dir.exists():
        for file_path in analysis_dir.glob('*'):
             if file_path.suffix in ['.png', '.json']:
                artifacts_list[file_path.name] = str(file_path)

    # Call unified LLM query handler
    answer = analyze_and_answer_query(
        query, 
        state["user_df"], 
        metrics, 
        artifacts_list, 
        state["target_column"], 
        state["task_type"]
    )
    
    return {"answer": answer, "plot_url": None}

# --- API Endpoints ---

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a CSV file and saves it to disk.
    Returns basic file information without LLM analysis.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    try:
        # Save uploaded file
        file_path = DATA_DIR / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        print(f"\n✓ File uploaded: {file.filename}")
        print(f"✓ Saved to: {file_path}\n")
        
        return {
            "filename": file.filename,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/api/analyze")
async def analyze_dataset(request: AnalysisRequest):
    """
    Triggers ML analysis with LLM-generated code.
    LLM generates complete pipeline code which is then executed.
    """
    from .llm_analysis_advisor import get_analysis_recommendations
    from .llm_pipeline_advisor import get_pipeline_strategy
    from .llm_code_generator import generate_ml_pipeline_code
    from .code_executor import execute_generated_code
    
    filename = request.filename
    target_column = request.target_column
    
    # Load dataset
    dataset_path = DATA_DIR / filename
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {filename}")
    
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    
    # Get LLM recommendations if no target specified
    if not target_column or target_column == "":
        print("\n" + "="*80)
        print("🎯 NO TARGET COLUMN SPECIFIED - REQUESTING LLM RECOMMENDATIONS")
        print("="*80 + "\n")
        
        recommendations = get_analysis_recommendations(df)
        
        # Use first suggested target
        if recommendations['target_columns']:
            target_column = recommendations['target_columns'][0]
            model_name = recommendations['recommended_model']
            model_type = recommendations['model_type']
            
            print(f"\n✓ Using LLM-suggested target: {target_column}")
            print(f"✓ Recommended model: {model_name}")
            print(f"✓ Model type: {model_type}\n")
            
        else:
            raise HTTPException(status_code=400, detail="Could not determine target column")
    else:
        # If target is provided, still get recommendations for model selection
        recommendations = get_analysis_recommendations(df)
        model_name = recommendations['recommended_model']
        model_type = recommendations['model_type']
    
    # Get LLM pipeline strategy
    print("="*80)
    print("🔧 REQUESTING PREPROCESSING STRATEGY")
    print("="*80 + "\n")
    
    pipeline_strategy = get_pipeline_strategy(df, target_column, model_name, model_type)
    
    print(f"\n✓ Pipeline strategy received\n")
    
    # Prepare dataset info for code generation
    dataset_info = {
        'shape': df.shape,
        'columns': {}
    }
    for col in df.columns:
        dataset_info['columns'][col] = {
            'dtype': str(df[col].dtype),
            'unique_count': int(df[col].nunique()),
            'is_numeric': pd.api.types.is_numeric_dtype(df[col])
        }
    
    # Generate ML pipeline code with LLM
    print("="*80)
    print("💻 GENERATING ML PIPELINE CODE")
    print("="*80 + "\n")
    
    generated_code = generate_ml_pipeline_code(
        filename, target_column, model_name, model_type,
        dataset_info, pipeline_strategy
    )
    
    # Execute generated code
    print("="*80)
    print("🚀 EXECUTING GENERATED CODE")
    print("="*80 + "\n")
    
    execution_result = execute_generated_code(generated_code)
    
    if not execution_result['success']:
        raise HTTPException(
            status_code=500,
            detail=f"Code execution failed: {execution_result['error']}"
        )
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    analysis_path = ARTIFACTS_DIR / analysis_id
    analysis_path.mkdir(exist_ok=True)

    # Move generated artifacts to analysis folder to isolate session
    final_artifacts = {}
    for key, path_str in execution_result['artifacts'].items():
        src_path = Path(path_str)
        if src_path.exists():
            dst_path = analysis_path / src_path.name
            # Copy or move
            import shutil
            shutil.copy2(src_path, dst_path)
            final_artifacts[key] = str(dst_path)

    # Save state for query handler
    # We need to reconstruct the artifact object or similar state
    # Since we don't have the explicit 'artifact' object from part3_ml_xai here (we used LLM code),
    # we'll save the dictionary of context needed for querying.
    state_to_save = {
        "user_df": df,
        "target_column": target_column,
        "task_type": model_type, # approximating task_type from model_type
        # Add a dummy artifact object or just dict if query_handler supports it
        # query_handler uses: state["user_df"], state["target_column"], state["task_type"]
        # It DOES NOT strictly use 'artifact' object anymore for the unified handler, just the artifacts list provided by fs
    }
    with open(analysis_path / "state.pkl", "wb") as f:
        pickle.dump(state_to_save, f)
        
    # Also move metrics.json if it exists
    metrics_src = ARTIFACTS_DIR / "metrics.json"
    if metrics_src.exists():
        shutil.copy2(metrics_src, analysis_path / "metrics.json")
    
    # Return results with artifact paths
    # Return results with artifact paths
    plot_urls = {}
    for key, path_str in final_artifacts.items():
        filename = Path(path_str).name
        plot_urls[key] = f"/artifacts/{analysis_id}/{filename}"

    return {
        "message": "Analysis complete",
        "analysis_id": analysis_id,
        "plots": plot_urls,
        "model": model_name,
        "target": target_column,
        "output": execution_result['output']
    }

@app.post("/api/query")
def query_endpoint(request: QueryRequest):
    """Handles user questions about a completed analysis."""
    try:
        result = handle_query(request.query, request.analysis_id)
        return result
    except Exception as e:
        print(f"ERROR in /api/query: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# --- Serve Frontend and Static Files ---

# Serve the generated artifacts (plots)
app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
# Serve the main frontend application
app.mount("/", StaticFiles(directory="src/explainable_dbms/xai_dbms_frontend/build", html=True), name="static")

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    # Open the web browser automatically after a short delay
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()

    # Run the server
    uvicorn.run(app, host=host, port=port)