"""
LLM-based ML Pipeline Advisor.

This module uses Gemini LLM to suggest intelligent preprocessing strategies
and model parameters based on dataset characteristics.
"""
from __future__ import annotations
import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Any


def get_pipeline_strategy(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    model_type: str
) -> Dict[str, Any]:
    """
    Uses LLM to suggest optimal preprocessing and training strategy.
    
    Args:
        df: The pandas DataFrame
        target_column: Target column name
        model_name: Recommended model (e.g., 'RandomForestRegressor')
        model_type: 'classification' or 'regression'
    
    Returns:
        Dictionary containing:
        - preprocessing: Preprocessing strategy
        - model_params: Model hyperparameters
        - split_ratio: Train/test split ratio
        - feature_engineering: Feature engineering suggestions
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return _fallback_pipeline_strategy(df, target_column, model_type)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Prepare dataset information
        dataset_info = _prepare_dataset_info(df, target_column)
        
        # Create prompt
        prompt = _create_pipeline_prompt(dataset_info, target_column, model_name, model_type)
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("🔧 REQUESTING ML PIPELINE STRATEGY FROM LLM")
        print("="*80)
        print(prompt)
        print("="*80 + "\n")
        
        print("⏳ Waiting for LLM pipeline strategy...")
        response = model.generate_content(prompt)
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("✅ RECEIVED PIPELINE STRATEGY")
        print("="*80)
        print(response.text)
        print("="*80 + "\n")
        
        result = _parse_pipeline_response(response.text)
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("📋 PIPELINE STRATEGY SUMMARY")
        print("="*80)
        print(f"✓ Missing Value Strategy: {result['preprocessing']['handle_missing']}")
        print(f"✓ Categorical Encoding: {result['preprocessing']['encode_categorical']}")
        print(f"✓ Feature Scaling: {result['preprocessing']['scale_features']}")
        print(f"✓ Train/Test Split: {result['split_ratio']}")
        print(f"✓ Model Parameters: {result['model_params']}")
        print("="*80 + "\n")
        
        return result
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR getting pipeline strategy: {e}")
        print("🔄 Using fallback strategy")
        print("="*80 + "\n")
        return _fallback_pipeline_strategy(df, target_column, model_type)


def _prepare_dataset_info(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Prepare comprehensive dataset information for LLM."""
    info = {
        'shape': df.shape,
        'columns': {},
        'target': target_column,
        'missing_values': df.isnull().sum().to_dict(),
        'sample_data': df.head(3).to_dict()
    }
    
    for col in df.columns:
        info['columns'][col] = {
            'dtype': str(df[col].dtype),
            'unique_count': int(df[col].nunique()),
            'null_count': int(df[col].isnull().sum()),
            'is_numeric': pd.api.types.is_numeric_dtype(df[col])
        }
    
    return info


def _create_pipeline_prompt(
    dataset_info: Dict[str, Any],
    target_column: str,
    model_name: str,
    model_type: str
) -> str:
    """Create prompt for pipeline strategy."""
    
    columns_summary = "\n".join([
        f"- {col}: {info['dtype']}, {info['unique_count']} unique, "
        f"{info['null_count']} nulls, {'numeric' if info['is_numeric'] else 'categorical'}"
        for col, info in dataset_info['columns'].items()
    ])
    
    prompt = f"""You are an ML pipeline expert. Suggest optimal preprocessing and training strategy.

Dataset Info:
- Shape: {dataset_info['shape']}
- Target Column: {target_column}
- Model: {model_name}
- Task: {model_type}

Columns:
{columns_summary}

Missing Values:
{json.dumps({k: v for k, v in dataset_info['missing_values'].items() if v > 0}, indent=2)}

Task: Suggest preprocessing strategy and model parameters.

Respond ONLY with valid JSON in this exact format:
{{
    "preprocessing": {{
        "handle_missing": "drop_rows|mean_impute|median_impute|mode_impute",
        "encode_categorical": ["one_hot"|"label_encode"],
        "scale_features": true|false,
        "feature_selection": true|false
    }},
    "split_ratio": 0.8,
    "model_params": {{
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }},
    "reasoning": "Brief explanation of strategy"
}}

Do not include any text before or after the JSON."""
    
    return prompt


def _parse_pipeline_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM's pipeline strategy response."""
    try:
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])
        
        parsed = json.loads(response_text)
        
        return {
            'preprocessing': parsed.get('preprocessing', {}),
            'split_ratio': parsed.get('split_ratio', 0.8),
            'model_params': parsed.get('model_params', {}),
            'reasoning': parsed.get('reasoning', '')
        }
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse pipeline response: {e}")
        return _default_pipeline_strategy()


def _fallback_pipeline_strategy(
    df: pd.DataFrame,
    target_column: str,
    model_type: str
) -> Dict[str, Any]:
    """Fallback pipeline strategy when LLM unavailable."""
    print("\n" + "="*80)
    print("🔧 USING FALLBACK PIPELINE STRATEGY")
    print("="*80)
    
    # Simple heuristics
    has_missing = df.isnull().sum().sum() > 0
    has_categorical = any(df[col].dtype == 'object' for col in df.columns if col != target_column)
    
    strategy = {
        'preprocessing': {
            'handle_missing': 'drop_rows' if has_missing else 'none',
            'encode_categorical': ['one_hot'] if has_categorical else [],
            'scale_features': True,
            'feature_selection': False
        },
        'split_ratio': 0.8,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'reasoning': 'Fallback: Standard preprocessing with default parameters'
    }
    
    print(f"✓ Strategy: {strategy['reasoning']}")
    print("="*80 + "\n")
    
    return strategy


def _default_pipeline_strategy() -> Dict[str, Any]:
    """Default pipeline strategy."""
    return {
        'preprocessing': {
            'handle_missing': 'drop_rows',
            'encode_categorical': ['one_hot'],
            'scale_features': True,
            'feature_selection': False
        },
        'split_ratio': 0.8,
        'model_params': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        'reasoning': 'Default strategy'
    }
