"""
LLM-based Analysis Advisor for comprehensive ML recommendations.

This module uses Gemini LLM to analyze datasets and provide:
1. Suggested target columns for prediction
2. Recommended ML model type (classification/regression)
3. Specific algorithm recommendation
4. Detailed reasoning for recommendations
"""
from __future__ import annotations
import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Any


def get_analysis_recommendations(
    df: pd.DataFrame,
    num_preview_rows: int = 10
) -> Dict[str, Any]:
    """
    Analyzes a DataFrame using LLM and provides comprehensive ML recommendations.
    
    Args:
        df: The pandas DataFrame to analyze
        num_preview_rows: Number of rows to include in preview (default: 10)
    
    Returns:
        Dictionary containing:
        - target_columns: List of suggested target column names
        - model_type: 'classification' or 'regression'
        - recommended_model: Specific algorithm name
        - reasoning: LLM's explanation
        - all_columns: List of all column names
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return _fallback_recommendations(df)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Prepare dataset information
        column_info = _extract_column_metadata(df)
        df_preview = df.head(num_preview_rows).to_string()
        
        # Create comprehensive analysis prompt
        prompt = _create_analysis_prompt(column_info, df_preview)
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("🤖 SENDING ANALYSIS REQUEST TO GEMINI LLM")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        print("⏳ Waiting for LLM analysis...")
        response = model.generate_content(prompt)
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("✅ RECEIVED LLM ANALYSIS")
        print("="*80)
        print(response.text)
        print("="*80 + "\n")
        
        result = _parse_analysis_response(response.text, df.columns.tolist())
        result['all_columns'] = df.columns.tolist()
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("📊 LLM ANALYSIS RECOMMENDATIONS")
        print("="*80)
        print(f"✓ Target Columns: {result['target_columns']}")
        print(f"✓ Model Type: {result['model_type']}")
        print(f"✓ Recommended Model: {result['recommended_model']}")
        print(f"💡 Reasoning: {result['reasoning']}")
        print("="*80 + "\n")
        
        return result
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR during LLM analysis: {e}")
        print("🔄 Falling back to heuristic-based recommendations")
        print("="*80 + "\n")
        return _fallback_recommendations(df)


def _extract_column_metadata(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract detailed metadata about each column."""
    column_info = []
    
    for col in df.columns:
        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'unique_count': int(df[col].nunique()),
            'sample_values': df[col].dropna().head(3).tolist()
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            info['is_numeric'] = True
            info['min'] = float(df[col].min()) if not df[col].isnull().all() else None
            info['max'] = float(df[col].max()) if not df[col].isnull().all() else None
            info['mean'] = float(df[col].mean()) if not df[col].isnull().all() else None
        else:
            info['is_numeric'] = False
        
        column_info.append(info)
    
    return column_info


def _create_analysis_prompt(
    column_info: List[Dict[str, Any]],
    df_preview: str
) -> str:
    """Create a comprehensive prompt for ML analysis recommendations."""
    
    column_summary = "\n".join([
        f"- {col['name']}: {col['dtype']}, "
        f"{col['unique_count']} unique values, "
        f"{'numeric' if col['is_numeric'] else 'categorical'}, "
        f"sample: {col['sample_values'][:3]}"
        for col in column_info
    ])
    
    prompt = f"""You are a machine learning expert. Analyze this dataset and provide comprehensive recommendations.

Dataset Preview:
{df_preview}

Column Information:
{column_summary}

Task: Provide ML recommendations including:
1. **Target Columns**: 1-3 best columns for prediction (outcomes, dependent variables)
2. **Model Type**: Classification or Regression
3. **Recommended Model**: Specific algorithm (e.g., RandomForestRegressor, XGBClassifier, LogisticRegression)
4. **Reasoning**: Why these recommendations

Consider:
- Column data types and distributions
- Business value (sales, revenue, churn, etc.)
- Suitability for ML tasks
- Data quality and completeness

Respond ONLY with valid JSON in this exact format:
{{
    "target_columns": ["column1", "column2"],
    "model_type": "regression",
    "recommended_model": "RandomForestRegressor",
    "reasoning": "Brief explanation of recommendations"
}}

Do not include any text before or after the JSON."""
    
    return prompt


def _parse_analysis_response(response_text: str, all_columns: List[str]) -> Dict[str, Any]:
    """Parse the LLM's JSON response and validate."""
    try:
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])
        
        # Parse JSON
        parsed = json.loads(response_text)
        
        # Validate target columns exist
        suggested = parsed.get('target_columns', [])
        valid_targets = [col for col in suggested if col in all_columns]
        
        return {
            'target_columns': valid_targets if valid_targets else suggested[:1],
            'model_type': parsed.get('model_type', 'regression').lower(),
            'recommended_model': parsed.get('recommended_model', 'RandomForestRegressor'),
            'reasoning': parsed.get('reasoning', 'LLM provided recommendations.')
        }
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response text: {response_text}")
        
        # Fallback: extract from text
        return {
            'target_columns': [all_columns[-1]] if all_columns else [],
            'model_type': 'regression',
            'recommended_model': 'RandomForestRegressor',
            'reasoning': 'Failed to parse LLM response, using defaults.'
        }


def _fallback_recommendations(df: pd.DataFrame) -> Dict[str, Any]:
    """Fallback heuristic-based recommendations when LLM unavailable."""
    print("\n" + "="*80)
    print("🔧 USING FALLBACK HEURISTIC RECOMMENDATIONS")
    print("="*80)
    
    all_columns = df.columns.tolist()
    
    # Find numeric columns for regression
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            if 0.01 < unique_ratio < 0.9 or df[col].nunique() > 20:
                numeric_cols.append(col)
    
    # Determine model type and target
    if numeric_cols:
        target = numeric_cols[0]
        model_type = 'regression'
        model = 'RandomForestRegressor'
    else:
        # Classification fallback
        for col in df.columns:
            if 2 <= df[col].nunique() <= 50:
                target = col
                model_type = 'classification'
                model = 'RandomForestClassifier'
                break
        else:
            target = all_columns[-1] if all_columns else 'unknown'
            model_type = 'regression'
            model = 'RandomForestRegressor'
    
    result = {
        'target_columns': [target],
        'model_type': model_type,
        'recommended_model': model,
        'reasoning': 'Heuristic: selected numeric column with good cardinality for regression.',
        'all_columns': all_columns
    }
    
    print(f"✓ Target: {result['target_columns']}")
    print(f"✓ Model Type: {result['model_type']}")
    print(f"✓ Model: {result['recommended_model']}")
    print("="*80 + "\n")
    
    return result
