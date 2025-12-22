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
    Answers a user query using an LLM with comprehensive analysis context.
    Provides rich dataset statistics, correlations, and insights.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-2.5-flash')

    # ========== BUILD COMPREHENSIVE CONTEXT ==========
    
    # 1. Dataset Overview
    dataset_info = f"""
DATASET OVERVIEW:
- Total Rows: {len(user_df)}
- Total Columns: {len(user_df.columns)}
- Target Column: {target_column}
- Task Type: {task_type}
"""
    
    # 2. Column Information with Statistics
    column_details = "\nCOLUMN DETAILS:\n"
    for col in user_df.columns:
        col_info = f"\n{col}:"
        col_info += f"\n  - Type: {user_df[col].dtype}"
        col_info += f"\n  - Non-null: {user_df[col].count()} / {len(user_df)}"
        col_info += f"\n  - Unique values: {user_df[col].nunique()}"
        
        if pd.api.types.is_numeric_dtype(user_df[col]):
            col_info += f"\n  - Mean: {user_df[col].mean():.4f}"
            col_info += f"\n  - Std: {user_df[col].std():.4f}"
            col_info += f"\n  - Min: {user_df[col].min():.4f}"
            col_info += f"\n  - Max: {user_df[col].max():.4f}"
            col_info += f"\n  - Median: {user_df[col].median():.4f}"
        else:
            # For categorical columns, show value counts
            value_counts = user_df[col].value_counts().head(10)
            col_info += f"\n  - Top values:\n"
            for val, count in value_counts.items():
                percentage = (count / len(user_df)) * 100
                col_info += f"    * {val}: {count} ({percentage:.1f}%)\n"
        
        column_details += col_info
    
    # 3. Target Variable Analysis
    target_analysis = f"\nTARGET VARIABLE ANALYSIS ({target_column}):\n"
    if pd.api.types.is_numeric_dtype(user_df[target_column]):
        target_analysis += f"- Mean: {user_df[target_column].mean():.4f}\n"
        target_analysis += f"- Std: {user_df[target_column].std():.4f}\n"
        target_analysis += f"- Min: {user_df[target_column].min():.4f}\n"
        target_analysis += f"- Max: {user_df[target_column].max():.4f}\n"
        target_analysis += f"- Median: {user_df[target_column].median():.4f}\n"
    else:
        target_counts = user_df[target_column].value_counts()
        target_analysis += "- Distribution:\n"
        for val, count in target_counts.items():
            percentage = (count / len(user_df)) * 100
            target_analysis += f"  * {val}: {count} ({percentage:.1f}%)\n"
    
    # 4. Correlations with Target (for numeric features)
    correlations = "\nFEATURE CORRELATIONS WITH TARGET:\n"
    numeric_cols = user_df.select_dtypes(include=['number']).columns.tolist()
    if target_column in numeric_cols and len(numeric_cols) > 1:
        target_corr = user_df[numeric_cols].corr()[target_column].sort_values(ascending=False)
        correlations += "Top correlations:\n"
        for feat, corr_val in target_corr.items():
            if feat != target_column:
                correlations += f"  - {feat}: {corr_val:.4f}\n"
    else:
        correlations += "(Correlation analysis not applicable for categorical target)\n"
    
    # 5. Cross-tabulations for categorical features (if target is categorical)
    crosstab_analysis = ""
    if not pd.api.types.is_numeric_dtype(user_df[target_column]):
        crosstab_analysis = "\nCROSS-TABULATION ANALYSIS:\n"
        categorical_cols = user_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
            if col != target_column and user_df[col].nunique() <= 10:
                crosstab = pd.crosstab(user_df[col], user_df[target_column], normalize='index') * 100
                crosstab_analysis += f"\n{col} vs {target_column} (% distribution):\n"
                crosstab_analysis += crosstab.to_string() + "\n"
    
    # 6. Sample Data
    sample_data = f"\nSAMPLE DATA (First 10 rows):\n{user_df.head(10).to_string()}\n"
    
    # 7. Model Performance
    model_performance = f"\nMODEL PERFORMANCE METRICS:\n{json.dumps(metrics, indent=2)}\n"
    
    # 8. Generated Artifacts
    artifacts_list = "\nGENERATED VISUALIZATIONS:\n"
    artifacts_list += "\n".join([f"- {name}" for name in artifacts.keys()])
    artifacts_list += "\n\nThese visualizations include:"
    artifacts_list += "\n- SHAP summary plot: Shows global feature importance"
    artifacts_list += "\n- LIME explanation plots: Show individual prediction explanations"
    
    # ========== CONSTRUCT PROMPT ==========
    
    prompt = f"""You are an expert Data Science Assistant with deep knowledge of machine learning and data analysis.

You have performed a complete ML analysis on a dataset. Use ALL the provided context to answer the user's question with detailed insights.

**USER QUESTION:** {query}

{dataset_info}

{column_details}

{target_analysis}

{correlations}

{crosstab_analysis}

{sample_data}

{model_performance}

{artifacts_list}

**INSTRUCTIONS:**
1. Analyze the comprehensive data provided above
2. Answer the user's question with specific insights from the data
3. Use actual statistics, percentages, and correlations in your answer
4. If asked about relationships or patterns, reference the cross-tabulations and correlations
5. If asked about feature importance, explain based on the SHAP visualizations mentioned
6. Provide actionable insights and explanations
7. Be specific and data-driven in your response

**ANSWER:**"""

    print("\n" + "="*80)
    print("❓ LLM QUERY")
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
