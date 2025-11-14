"""
Command-line entry point to run the full explainable DBMS pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import get_database_config, get_paths_config
from .part1_setup import initialize_database
from .part2_schema_data import load_user_data, compute_aggregated_features
from .part3_ml_xai import ModelArtifact, train_models_and_explain
from .part4_visualization import generate_visualizations, generate_shap_force_plot_for_instance
from .llm_summarizer import summarize_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def answer_user_query(query: str, artifacts: List[ModelArtifact], feature_df: pd.DataFrame, target_column: str) -> None:
    """
    Answers a user query with visualizations and LLM summaries.
    """
    logger.info(f"Answering user query: {query}")

    match = re.search(r"customer (\d+)", query, re.IGNORECASE)
    if not match:
        print("Sorry, I can only answer questions about specific customers, e.g., 'why did customer 123 churn?'")
        return

    customer_id = int(match.group(1))
    artifact = artifacts[0]

    if artifact.shap_sample.empty:
        print("Sorry, no SHAP explanations were generated. Cannot answer the query.")
        return

    try:
        instance_index = feature_df[feature_df['customer_id'] == customer_id].index[0]
        if instance_index not in artifact.shap_sample.index:
            print(f"Sorry, customer {customer_id} was not in the sample used for explanations.")
            return
        
        force_plot = generate_shap_force_plot_for_instance(artifact, instance_index)
        plt.show()

        shap_values = artifact.shap_values[instance_index]
        feature_names = artifact.shap_sample.columns
        
        explanation_text = f"Explanation for customer {customer_id}:\n"
        for feature, shap_value in zip(feature_names, shap_values):
            explanation_text += f"- {feature}: {shap_value:.4f}\n"

        summary = summarize_text(explanation_text)
        print("\nSummary of the prediction:")
        print(summary)

    except IndexError:
        print(f"Sorry, I couldn't find customer {customer_id} in the dataset.")
    except Exception as e:
        logger.error(f"An error occurred while answering the query: {e}")


def run_pipeline() -> None:
    """Execute the complete pipeline end-to-end."""
    logger.info("Initialising configuration.")
    db_config = get_database_config()
    paths_config = get_paths_config()

    logger.info("Setting up database and schema.")
    engine = initialize_database(db_config)

    try:
        file_path = "e:\\github_projects\\explainable_dbms\\outputs\\dummy_data.csv"
        target_column = "churn"
        
        table_name = Path(file_path).stem
        logger.info(f"Loading user data from {file_path} into table {table_name}.")
        user_df = load_user_data(engine, file_path, paths_config)

        if target_column not in user_df.columns:
            logger.error(f"Target column '{target_column}' not found in the dataset.")
            return

        logger.info("Computing aggregated analytical features.")
        feature_df = compute_aggregated_features(engine, table_name, paths_config)

        logger.info("Training models and generating explanations.")
        artifacts: List[ModelArtifact] = train_models_and_explain(engine, feature_df, target_column, paths_config)

        logger.info("Creating visualization dashboards.")
        generate_visualizations(artifacts, paths_config)

        logger.info("Model training and initial analysis complete.")
        
        # This is commented out for testing purposes
        # while True:
        #     user_query = input("Ask a question about your data (or type 'exit' to quit): ")
        #     if user_query.lower() == 'exit':
        #         break
        #     answer_user_query(user_query, artifacts, feature_df, target_column)

    except Exception as e:
        logger.error(f"An error occurred during the pipeline execution: {e}")
        return


if __name__ == "__main__":
    run_pipeline()