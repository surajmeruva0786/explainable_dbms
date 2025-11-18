import uvicorn
import webbrowser
import threading
import pickle
import uuid
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import re

from .part1_setup import initialize_database
from .part2_schema_data import load_user_data, compute_aggregated_features
from .part3_ml_xai import train_models_and_explain
from .part4_visualization import generate_visualizations
from .query_handler import answer_user_query, answer_general_query
from .llm_model_selector import select_model_with_llm

# --- Directory setup for state management ---
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("temp_data")
DATA_DIR.mkdir(exist_ok=True)

# --- FastAPI App and Data Models ---
app = FastAPI()

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

    if re.search(r"explain Rank (\d+)", query, re.IGNORECASE):
        answer, plot = answer_user_query(query, [state["artifact"]], state["feature_df"], state["target_column"], state["task_type"])
        plot_path = ARTIFACTS_DIR / analysis_id / "temp_explanation.png"
        plot.savefig(plot_path)
        plot_url = f"/artifacts/{analysis_id}/temp_explanation.png"
        return {"answer": answer, "plot_url": plot_url}
    else:
        answer = answer_general_query(query, state["user_df"])
        return {"answer": answer, "plot_url": None}

# --- API Endpoints ---

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Handles CSV file upload, saves it, and returns its columns."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV.")
    
    file_path = DATA_DIR / file.filename
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    df = pd.read_csv(file_path)
    return {"filename": file.filename, "columns": df.columns.tolist()}

@app.post("/api/analyze")
def analyze_endpoint(request: AnalysisRequest):
    """Triggers the main analysis pipeline."""
    try:
        result = run_analysis(request.filename, request.target_column)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_endpoint(request: QueryRequest):
    """Handles user questions about a completed analysis."""
    try:
        result = handle_query(request.query, request.analysis_id)
        return result
    except Exception as e:
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