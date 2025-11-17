"""
This module uses an LLM to select the best model for a given dataset and target.
"""
from __future__ import annotations
import json
from .llm_summarizer import summarize_text

def select_model_with_llm(df_head: str, target_column: str, task_type: str) -> str:
    """
    Uses an LLM to select the best model for a given dataset and target.

    Args:
        df_head: The head of the dataframe as a string.
        target_column: The name of the target column.
        task_type: The type of task (classification or regression).

    Returns:
        The name of the selected model.
    """
    if task_type == "classification":
        models = ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"]
    else:
        models = ["RandomForestRegressor", "GradientBoostingRegressor", "XGBRegressor"]

    prompt = f"""
    Given the following dataset head and target column, which model would be the best to train?
    Dataset Head:
    {df_head}

    Target Column: {target_column}
    Task Type: {task_type}

    Available Models: {', '.join(models)}

    Please choose one model from the list above. Your response should be only the name of the model.
    """
    print("--- LLM Prompt ---")
    print(prompt)

    # This is a placeholder for a real LLM call.
    # In a real application, you would use a library like langchain or openai
    # to interact with an LLM. For now, we'll just pick the first model.
    # selected_model = summarize_text(prompt) 
    
    # For now, let's just return the first model in the list.
    selected_model = models[0]
    print(f"--- Selected Model ---")
    print(selected_model)
    
    return selected_model
