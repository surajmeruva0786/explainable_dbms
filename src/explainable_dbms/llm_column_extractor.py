"""
LLM-based intelligent column extraction for target column selection.

This module uses Gemini LLM to analyze uploaded datasets and suggest
the most appropriate target columns for machine learning analysis.
"""
from __future__ import annotations
import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, List, Any


def extract_target_columns_with_llm(
    df: pd.DataFrame, 
    num_preview_rows: int = 5
) -> Dict[str, Any]:
    """
    Analyzes a DataFrame using LLM and suggests appropriate target columns.
    
    Args:
        df: The pandas DataFrame to analyze
        num_preview_rows: Number of rows to include in the preview (default: 5)
    
    Returns:
        Dictionary containing:
        - all_columns: List of all column names
        - suggested_targets: List of suggested target column names
        - reasoning: LLM's explanation for the suggestions
        - column_info: Metadata about each column
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        # Fallback: return all numeric columns as suggestions
        return _fallback_column_extraction(df)
    
    try:
        genai.configure(api_key=api_key)
        # Use models/gemini-2.5-flash - confirmed working model
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Prepare dataset information for LLM
        column_info = _extract_column_metadata(df)
        df_preview = df.head(num_preview_rows).to_string()
        
        # Create structured prompt for LLM
        prompt = _create_column_extraction_prompt(column_info, df_preview)
        
        # VERBOSE LOGGING - Show prompt being sent to LLM
        print("\n" + "="*80)
        print("🤖 SENDING PROMPT TO GEMINI LLM")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        # Get LLM response
        print("⏳ Waiting for LLM response...")
        response = model.generate_content(prompt)
        
        # VERBOSE LOGGING - Show raw LLM response
        print("\n" + "="*80)
        print("✅ RECEIVED LLM RESPONSE")
        print("="*80)
        print(response.text)
        print("="*80 + "\n")
        
        result = _parse_llm_response(response.text, df.columns.tolist())
        
        # Add column metadata to result
        result['column_info'] = column_info
        result['all_columns'] = df.columns.tolist()
        
        # VERBOSE LOGGING - Show parsed result
        print("\n" + "="*80)
        print("📊 LLM COLUMN EXTRACTION RESULT")
        print("="*80)
        print(f"✓ All Columns: {result['all_columns']}")
        print(f"⭐ Suggested Targets: {result['suggested_targets']}")
        print(f"💡 Reasoning: {result['reasoning']}")
        print("="*80 + "\n")
        
        return result
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR during LLM column extraction: {e}")
        print("🔄 Falling back to heuristic-based extraction")
        print("="*80 + "\n")
        return _fallback_column_extraction(df)


def _extract_column_metadata(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Extract metadata about each column in the DataFrame."""
    column_info = []
    
    for col in df.columns:
        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'unique_count': int(df[col].nunique()),
            'sample_values': df[col].dropna().head(3).tolist()
        }
        
        # Add statistics for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            info['is_numeric'] = True
            info['min'] = float(df[col].min()) if not df[col].isnull().all() else None
            info['max'] = float(df[col].max()) if not df[col].isnull().all() else None
            info['mean'] = float(df[col].mean()) if not df[col].isnull().all() else None
        else:
            info['is_numeric'] = False
        
        column_info.append(info)
    
    return column_info


def _create_column_extraction_prompt(
    column_info: List[Dict[str, Any]], 
    df_preview: str
) -> str:
    """Create a structured prompt for the LLM to analyze columns."""
    
    column_summary = "\n".join([
        f"- {col['name']}: {col['dtype']}, "
        f"{col['unique_count']} unique values, "
        f"{'numeric' if col['is_numeric'] else 'categorical'}, "
        f"sample: {col['sample_values'][:3]}"
        for col in column_info
    ])
    
    prompt = f"""You are a data science expert analyzing a dataset to identify the best target column(s) for machine learning.

Dataset Preview:
{df_preview}

Column Information:
{column_summary}

Task: Identify 1-3 columns that would be most suitable as target variables for machine learning analysis. Consider:
1. Columns that represent outcomes, predictions, or dependent variables
2. Numeric columns for regression tasks
3. Categorical columns with reasonable number of classes for classification
4. Columns with business value (e.g., sales, revenue, churn, risk)

Respond ONLY with valid JSON in this exact format:
{{
    "suggested_targets": ["column1", "column2"],
    "reasoning": "Brief explanation of why these columns were selected"
}}

Do not include any text before or after the JSON."""
    
    return prompt


def _parse_llm_response(response_text: str, all_columns: List[str]) -> Dict[str, Any]:
    """Parse the LLM's JSON response and validate column names."""
    try:
        # Try to extract JSON from the response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])
        
        # Parse JSON
        parsed = json.loads(response_text)
        
        # Validate that suggested columns exist in the dataset
        suggested = parsed.get('suggested_targets', [])
        valid_suggestions = [col for col in suggested if col in all_columns]
        
        return {
            'suggested_targets': valid_suggestions if valid_suggestions else suggested[:3],
            'reasoning': parsed.get('reasoning', 'LLM suggested these columns as targets.')
        }
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM response as JSON: {e}")
        print(f"Response text: {response_text}")
        
        # Fallback: try to extract column names from text
        suggested = [col for col in all_columns if col.lower() in response_text.lower()]
        return {
            'suggested_targets': suggested[:3] if suggested else [],
            'reasoning': 'Extracted from LLM text response.'
        }


def _fallback_column_extraction(df: pd.DataFrame) -> Dict[str, Any]:
    """Fallback method when LLM is unavailable - use heuristics."""
    print("\n" + "="*80)
    print("🔧 USING FALLBACK HEURISTIC METHOD")
    print("="*80)
    
    all_columns = df.columns.tolist()
    
    # Heuristic: suggest numeric columns with reasonable unique values
    suggested = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            # Good target: not too few unique values (not binary) and not too many (not ID)
            if 0.01 < unique_ratio < 0.9 or df[col].nunique() > 20:
                suggested.append(col)
                print(f"  ✓ Selected '{col}' (numeric, {df[col].nunique()} unique values)")
    
    # If no numeric columns, suggest columns with moderate cardinality
    if not suggested:
        for col in df.columns:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 50:
                suggested.append(col)
                print(f"  ✓ Selected '{col}' (categorical, {unique_count} unique values)")
    
    column_info = _extract_column_metadata(df)
    
    result = {
        'all_columns': all_columns,
        'suggested_targets': suggested[:3],  # Limit to top 3
        'reasoning': 'Fallback heuristic: selected numeric columns with appropriate cardinality.',
        'column_info': column_info
    }
    
    print(f"\n📊 Fallback Result:")
    print(f"  All Columns: {all_columns}")
    print(f"  Suggested: {result['suggested_targets']}")
    print("="*80 + "\n")
    
    return result
