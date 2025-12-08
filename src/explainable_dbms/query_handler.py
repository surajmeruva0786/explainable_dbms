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
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

def analyze_and_answer_query(
    query: str, 
    user_df: pd.DataFrame, 
    metrics: dict, 
    artifacts: dict,
    target_column: str,
    task_type: str
) -> str:
    """
    Answers a user query using an LLM with full analysis context.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-1.5-flash')

    # Construct context
    df_head = user_df.head().to_string()
    
    # Summarize visualizations
    viz_list = "\n".join([f"- {name}: {path}" for name, path in artifacts.items()])
    
    prompt = f"""You are an expert Data Science Assistant. You have performed an ML analysis on a dataset.
Answer the user's question based strictly on the provided context.

**User Query:** {query}

**Dataset Context:**
- Columns: {list(user_df.columns)}
- Target Column: {target_column}
- Task Type: {task_type}
- First 5 Rows:
{df_head}

**Model Performance:**
{json.dumps(metrics, indent=2)}

**Generated Visualizations:**
{viz_list}

**Instructions:**
1. Answer the user's question clearly and concisely.
2. If the user asks about model performance, cite the metrics.
3. If the user asks about specific instances, explain that LIME plots have been generated for the first 3 test instances.
4. If the user asks about global feature importance, mention the SHAP summary plot.
5. Do not hallucinate. If the answer is not in the context, say so.

**Answer:**"""

    print("\n" + "="*80)
    print("❓ LLM QUERY PROMPT")
    print("="*80)
    print(query)
    print("="*80 + "\n")

    try:
        response = model.generate_content(prompt)
        print("\n" + "="*80)
        print("💡 LLM RESPONSE")
        print("="*80)
        print(response.text)
        print("="*80 + "\n")
        return response.text
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        return f"Error generating answer: {e}"
