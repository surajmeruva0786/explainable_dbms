import gradio as gr
import pandas as pd
from pathlib import Path
import re

from .part1_setup import initialize_database
from .part2_schema_data import load_user_data, compute_aggregated_features
from .part3_ml_xai import train_models_and_explain
from .part4_visualization import generate_visualizations
from .query_handler import answer_user_query, answer_general_query
from .llm_model_selector import select_model_with_llm

def process_dataset(file):
    if file is None:
        return None, gr.update(choices=[])
    
    print("Processing dataset...")
    df = pd.read_csv(file.name)
    print("Dataset processed successfully.")
    return df, gr.update(choices=df.columns.tolist())

def run_analysis(df, target_column):
    if df is None or target_column is None:
        return "Please upload a dataset and select a target column.", None, None, None, None, None, None, None, None, None

    print("Starting analysis...")
    
    # 1. Setup
    print("Step 1: Initializing database...")
    engine = initialize_database()
    
    # Create a temporary path for the dataframe
    temp_path = Path("temp_dataset.csv")
    df.to_csv(temp_path, index=False)

    # 2. Load data
    print("Step 2: Loading data...")
    user_df = load_user_data(engine, str(temp_path), None)
    table_name = temp_path.stem

    # 3. Determine task type
    print("Step 3: Determining task type...")
    target_series = user_df[target_column]
    if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20:
        task_type = "regression"
    else:
        task_type = "classification"
    print(f"Task type: {task_type}")

    # 4. Feature Engineering
    print("Step 4: Computing aggregated features...")
    feature_df = compute_aggregated_features(engine, table_name, None)

    # 5. LLM Model Selection
    print("Step 5: Selecting model with LLM...")
    df_head = df.head().to_string()
    selected_model = select_model_with_llm(df_head, target_column, task_type)
    print(f"Selected model: {selected_model}")

    # 6. Model Training
    print("Step 6: Training model and explaining...")
    artifact = train_models_and_explain(engine, feature_df, target_column, task_type, selected_model, None)
    
    # 7. Visualization
    print("Step 7: Generating visualizations...")
    visualizations = generate_visualizations([artifact])
    plots = list(visualizations[artifact.model_name].values())
    print("Analysis complete.")

    return f"Analysis complete. Selected model: {artifact.model_name}. You can now ask questions.", artifact, user_df, feature_df, task_type, plots[0], plots[1], plots[2], plots[3], plots[4]

def handle_query(query, artifact, user_df, feature_df, target_column, task_type):
    if not query:
        return "Please enter a query.", None

    if re.search(r"explain Rank (\d+)", query, re.IGNORECASE):
        answer, plot = answer_user_query(query, [artifact], feature_df, target_column, task_type)
        return answer, plot
    else:
        answer = answer_general_query(query, user_df)
        return answer, None

with gr.Blocks() as demo:
    gr.Markdown("# Explainable DBMS")
    
    # State variables
    artifact_state = gr.State(None)
    user_df_state = gr.State(None)
    feature_df_state = gr.State(None)
    task_type_state = gr.State(None)
    target_column_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            file_upload = gr.File(label="Upload CSV")
            target_column_dropdown = gr.Dropdown(label="Select Target Column")
            run_button = gr.Button("Run Analysis")
            analysis_status = gr.Textbox(label="Analysis Status", interactive=False)
        with gr.Column(scale=2):
            gr.Markdown("## Analysis Visualizations")
            summary_plot = gr.Plot(label="SHAP Summary Plot")
            waterfall_plot = gr.Plot(label="SHAP Waterfall Plot")
            bar_plot = gr.Plot(label="SHAP Bar Plot")
            lime_plot = gr.Plot(label="LIME Plot")
            comparison_plot = gr.Plot(label="Comparison Plot")


    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(label="Ask a question")
            ask_button = gr.Button("Ask")
        with gr.Column(scale=2):
            query_output = gr.Textbox(label="Answer")
            explanation_plot = gr.Plot(label="Explanation Plot")

    # Define interactions
    file_upload.change(process_dataset, inputs=file_upload, outputs=[user_df_state, target_column_dropdown])
    
    run_button.click(
        run_analysis, 
        inputs=[user_df_state, target_column_dropdown], 
        outputs=[analysis_status, artifact_state, user_df_state, feature_df_state, task_type_state, summary_plot, waterfall_plot, bar_plot, lime_plot, comparison_plot]
    )
    
    target_column_dropdown.change(lambda x: x, inputs=target_column_dropdown, outputs=target_column_state)

    ask_button.click(
        handle_query,
        inputs=[query_input, artifact_state, user_df_state, feature_df_state, target_column_state, task_type_state],
        outputs=[query_output, explanation_plot]
    )


if __name__ == "__main__":
    demo.launch()