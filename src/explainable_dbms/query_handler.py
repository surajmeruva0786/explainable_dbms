"""
Query handling logic for the explainable DBMS.
"""

from __future__ import annotations

import re
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

from .part3_ml_xai import ModelArtifact
from .part4_visualization import generate_shap_force_plot_for_instance
from .llm_summarizer import summarize_text

def answer_user_query(query: str, artifacts: List[ModelArtifact], feature_df: pd.DataFrame, target_column: str, task_type: str) -> Tuple[str, Optional[plt.Figure]]:
    """
    Answers a user query with visualizations and LLM summaries.
    Returns a string with the explanation and a plot object.
    """
    match = re.search(r"explain index (\d+)", query, re.IGNORECASE)
    if not match:
        return "Sorry, I can only answer questions about specific indices, e.g., 'explain index 123'", None

    instance_id = int(match.group(1))
    artifact = artifacts[0]

    if artifact.shap_sample.empty:
        return "Sorry, no SHAP explanations were generated. Cannot answer the query.", None

    try:
        instance_index = feature_df.index.get_loc(instance_id)
        
        prediction_df = artifact.predictions
        prediction = prediction_df[prediction_df['instance_id'] == str(instance_id)].iloc[0]
        predicted_value = prediction['prediction_value']

        explanation_text = f"Explanation for index {instance_id}:\n"
        if task_type == "classification":
            predicted_class = prediction['predicted_class']
            explanation_text += f"The predicted class is '{predicted_class}' with a probability of {predicted_value:.4f}.\n"
        else:
            explanation_text += f"The predicted value for {target_column} is {predicted_value:.4f}.\n"

        if instance_index not in artifact.shap_sample.index:
            explanation_text += "\nA detailed SHAP explanation is not available for this instance as it was not in the explanation sample."
            summary = summarize_text(explanation_text)
            return summary, None

        force_plot = generate_shap_force_plot_for_instance(artifact, instance_index)
        
        shap_values = artifact.shap_values[instance_index]
        feature_names = artifact.shap_sample.columns
        
        explanation_text += "The main factors influencing this prediction are:\n"
        for feature, shap_value in zip(feature_names, shap_values):
            explanation_text += f"- {feature}: {shap_value:.4f}\n"

        summary = summarize_text(explanation_text)
        return summary, force_plot

    except IndexError:
        return f"Sorry, I couldn't find index {instance_id} in the dataset.", None
    except Exception as e:
        return f"An error occurred while answering the query: {e}", None


def answer_general_query(query: str, user_df: pd.DataFrame) -> str:
    """Answers a general query about the dataset using an LLM."""
    
    prompt = f"""
    You are a data analyst. A user has asked a question about a dataset.
    Answer the user's query based on the provided data.
    
    User Query: "{query}"
    
    Dataset columns: {user_df.columns.tolist()}
    
    First 5 rows of the dataset:
    {user_df.head().to_string()}
    
    Answer:
    """
    
    try:
        answer = summarize_text(prompt)
        return answer
    except Exception as e:
        return f"An error occurred while answering the general query: {e}"
