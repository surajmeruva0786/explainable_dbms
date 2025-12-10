"""
LLM Code Generator for ML Pipeline.

This module uses Gemini LLM to generate complete, executable Python code
for data preprocessing, model training, and XAI visualizations.
"""
from __future__ import annotations
import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Dict, Any


def generate_ml_pipeline_code(
    filename: str,
    target_column: str,
    model_name: str,
    model_type: str,
    dataset_info: Dict[str, Any],
    pipeline_strategy: Dict[str, Any]
) -> str:
    """
    Generates complete Python code for ML pipeline using LLM.
    
    Args:
        filename: CSV filename
        target_column: Target column name
        model_name: Model to use (e.g., 'RandomForestRegressor')
        model_type: 'classification' or 'regression'
        dataset_info: Dataset metadata
        pipeline_strategy: Preprocessing strategy from LLM
    
    Returns:
        Executable Python code as string
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return _fallback_code_template(filename, target_column, model_name, model_type)
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        
        # Create comprehensive prompt
        prompt = _create_code_generation_prompt(
            filename, target_column, model_name, model_type,
            dataset_info, pipeline_strategy
        )
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("💻 REQUESTING CODE GENERATION FROM LLM")
        print("="*80)
        print(prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
        print("="*80 + "\n")
        
        print("⏳ Waiting for LLM to generate code...")
        response = model.generate_content(prompt)
        
        # VERBOSE LOGGING
        print("\n" + "="*80)
        print("✅ RECEIVED GENERATED CODE")
        print("="*80)
        print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        print("="*80 + "\n")
        
        # Extract code from response
        code = _extract_code_from_response(response.text)
        
        print(f"\n✓ Code generated successfully ({len(code)} characters)")
        print(f"✓ Ready for execution\n")
        
        # Log to Firestore
        from .firestore_logger import log_llm_code_generation
        log_llm_code_generation(prompt, code, success=True)
        
        return code
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ ERROR generating code: {e}")
        # Log failure
        from .firestore_logger import log_llm_code_generation
        log_llm_code_generation("", "", success=False, error=str(e))
        
        print("🔄 Using fallback code template")
        print("="*80 + "\n")
        return _fallback_code_template(filename, target_column, model_name, model_type)


def _create_code_generation_prompt(
    filename: str,
    target_column: str,
    model_name: str,
    model_type: str,
    dataset_info: Dict[str, Any],
    pipeline_strategy: Dict[str, Any]
) -> str:
    """Create comprehensive prompt for code generation."""
    
    columns_summary = "\n".join([
        f"- {col}: {info['dtype']}, {info['unique_count']} unique, "
        f"{'numeric' if info['is_numeric'] else 'categorical'}"
        for col, info in dataset_info['columns'].items()
    ])
    
    prompt = f"""You are an expert ML engineer. Generate complete, executable Python code for an ML pipeline.

**Requirements:**

Dataset: temp_data/{filename}
Target Column: {target_column}
Model: {model_name}
Task Type: {model_type}

**Dataset Info:**
{columns_summary}

**Preprocessing Strategy:**
{json.dumps(pipeline_strategy['preprocessing'], indent=2)}

**Model Parameters:**
{json.dumps(pipeline_strategy['model_params'], indent=2)}

**Code Requirements:**

1. **Imports:** Include all necessary imports. EXPLICITLY import the model class (e.g., `from xgboost import XGBClassifier`).
2. **Load Data:** Load from temp_data/{filename}
3. **Preprocessing:**
   - Handle missing values: {pipeline_strategy['preprocessing'].get('handle_missing', 'drop_rows')}
   - Encode categorical: {pipeline_strategy['preprocessing'].get('encode_categorical', [])}
   - Scale features: {pipeline_strategy['preprocessing'].get('scale_features', False)}
4. **Split Data:** {pipeline_strategy.get('split_ratio', 0.8)} train/test split
5. **Train Model:** {model_name} with params: {pipeline_strategy['model_params']}
6. **Generate Visualizations:**
   - SHAP summary plot - save to artifacts/shap_summary.png
   - LIME explanation for first 3 predictions - save to artifacts/lime_explanation_1.png, artifacts/lime_explanation_2.png, artifacts/lime_explanation_3.png
7. **Save Results:** Save trained model metrics to artifacts/metrics.json

**IMPORTANT:**
- Ensure all data is numeric (float type) before SHAP/LIME. Convert categorical to numeric first using LabelEncoder.
- Drop any non-numeric columns if conversion fails.
- Generate 3 separate LIME plots, one for each of the first 3 test instances.

**IMPORTANT:** Ensure all data is numeric (float type) before SHAP/LIME. Convert categorical to numeric first.

**Output Format:**
- Return ONLY executable Python code
- No explanations, no markdown, no comments outside code
- Code must be production-ready
- Include error handling
- Use print statements for progress

Generate the complete code now:"""
    
    return prompt


def _extract_code_from_response(response_text: str) -> str:
    """Extract Python code from LLM response."""
    # Remove markdown code blocks if present
    if '```python' in response_text:
        # Extract code between ```python and ```
        start = response_text.find('```python') + 9
        end = response_text.find('```', start)
        code = response_text[start:end].strip()
    elif '```' in response_text:
        # Generic code block
        start = response_text.find('```') + 3
        end = response_text.find('```', start)
        code = response_text[start:end].strip()
    else:
        # Assume entire response is code
        code = response_text.strip()
    
    return code


def _fallback_code_template(
    filename: str,
    target_column: str,
    model_name: str,
    model_type: str
) -> str:
    """Fallback code template when LLM unavailable."""
    
    print("\n" + "="*80)
    print("🔧 USING FALLBACK CODE TEMPLATE")
    print("="*80 + "\n")
    
    code = f"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

Path('artifacts').mkdir(exist_ok=True)

print("1. Loading data...")
df = pd.read_csv('temp_data/{filename}')
print(f"Loaded: {{df.shape}}")

print("2. Preprocessing...")
df = df.dropna(subset=['{target_column}'])
X = df.drop(columns=['{target_column}'])
y = df['{target_column}']

# Encode categoricals
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Keep only numeric
X = X.select_dtypes(include=[np.number]).astype(float)
y = y.astype(float)

# Impute missing
if X.isnull().sum().sum() > 0:
    X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)

print("3. Splitting...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {{X_train.shape}}, Test: {{X_test.shape}}")

print("4. Training...")
model = {model_name}(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Score: {{score:.4f}}")

with open('artifacts/metrics.json', 'w') as f:
    json.dump({{'model': '{model_name}', 'score': float(score)}}, f)

print("5. SHAP...")
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:100])
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test[:100], show=False)
    plt.tight_layout()
    plt.savefig('artifacts/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP saved!")
except Exception as e:
    print(f"SHAP failed: {{e}}")

    # Ensure STRICTLY numeric for XAI
    X_train_num = X_train.select_dtypes(include=[np.number]).astype(float)
    X_test_num = X_test.select_dtypes(include=[np.number]).astype(float)
    
    # Fill remaining NaNs if any
    if X_train_num.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        X_train_num = pd.DataFrame(imputer.fit_transform(X_train_num), columns=X_train_num.columns)
        X_test_num = pd.DataFrame(imputer.transform(X_test_num), columns=X_test_num.columns)

    print("6. LIME...")
    try:
        lime_exp = LimeTabularExplainer(
            X_train_num.values, 
            feature_names=X_train_num.columns.tolist(), 
            mode='{"regression" if model_type == "regression" else "classification"}'
        )
        
        for i in range(3):
            try:
                # Use numeric data for explanation
                exp = lime_exp.explain_instance(
                    X_test_num.iloc[i].values, 
                    model.predict, 
                    num_features=5
                )
                
                # Save each plot separately
                plt.figure(figsize=(6, 4))
                exp.as_pyplot_figure()
                plt.title(f'LIME Explanation - Instance {{i+1}}')
                plt.tight_layout()
                plt.savefig(f'artifacts/lime_explanation_{{i+1}}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"LIME plot {{i+1}} saved!")
            except Exception as e:
                print(f"LIME instance {{i+1}} failed: {{e}}")
                
    except Exception as e:
        print(f"LIME initialization failed: {{e}}")

print("Pipeline complete!")
"""
    
    return code.strip()
