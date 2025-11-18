"""
Part 3: Machine learning training, prediction storage, and explainability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import time
from sqlalchemy.engine import Engine

from .config import PathsConfig, get_paths_config
from .io_utils import save_json
from .llm_summarizer import summarize_text, save_summary


@dataclass
class ModelArtifact:
    """Container for fitted models and explainability metadata."""

    model_name: str
    model: Any
    metrics: Dict[str, Any]
    feature_importance: Optional[List[float]]
    shap_values: np.ndarray
    shap_expected_value: Any
    shap_sample: pd.DataFrame
    lime_explanations: Dict[int, List[Tuple[str, float]]]
    predictions: pd.DataFrame


def prepare_datasets(feature_df: pd.DataFrame, target_column: str, task_type: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Split the aggregated feature dataframe into train/test sets."""
    
    # Use the first column as the index
    index_column = feature_df.columns[0]
    feature_df = feature_df.set_index(index_column)

    feature_columns = [col for col in feature_df.columns if col != target_column]
    
    numeric_feature_columns = feature_df[feature_columns].select_dtypes(include=np.number).columns.tolist()
    
    X = feature_df[numeric_feature_columns]
    y = feature_df[target_column]
    
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )

    imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    return X_train, X_test, y_train, y_test, numeric_feature_columns


def _train_random_forest_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)
    return model

def _train_random_forest_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _train_gradient_boosting_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model

def _train_gradient_boosting_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _train_xgboost_classifier(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)
    return model

def _train_xgboost_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _generate_shap_values(model: Any, X_train: pd.DataFrame, X_sample: pd.DataFrame) -> Tuple[np.ndarray, Any]:
    """Compute SHAP values for a sample of instances."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    expected_value = explainer.expected_value
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if isinstance(expected_value, list):
        expected_value = expected_value[-1]
    shap_array = np.atleast_2d(np.asarray(shap_values))
    return shap_array, expected_value


def _generate_lime_explanations(
    model: Any,
    X_train: pd.DataFrame,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    task_type: str,
    class_names: Optional[List[str]] = None
) -> Dict[int, List[Tuple[str, float]]]:
    """Generate LIME explanations with top contributing features."""
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode=task_type,
        discretize_continuous=True,
        verbose=False,
    )

    explanations: Dict[int, List[Tuple[str, float]]] = {}
    predict_fn = model.predict_proba if task_type == "classification" else model.predict
    for idx, row in X_sample.iterrows():
        exp = explainer.explain_instance(
            data_row=row.values,
            predict_fn=predict_fn,
            num_features=min(5, len(feature_names)),
        )
        explanations[idx] = exp.as_list()
    return explanations


def _compute_classification_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Generate standard classification metrics and predictions."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return metrics, y_pred, y_prob

def _compute_regression_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Generate standard regression metrics and predictions."""
    y_pred = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }
    return metrics, y_pred


def _assemble_predictions_df(
    instance_ids: pd.Index,
    model_name: str,
    target_column: str,
    task_type: str,
    y_pred: np.ndarray,
    shap_payload: Dict[int, Any],
    lime_payload: Dict[int, Any],
    y_prob: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Create a dataframe ready for persistence in the prediction_results table."""
    rows = []
    timestamp = datetime.utcnow()
    
    if task_type == "classification":
        for idx, probability, pred_class in zip(instance_ids, y_prob, y_pred):
            shap_json = json.dumps(shap_payload.get(idx)) if idx in shap_payload else None
            lime_json = json.dumps(lime_payload.get(idx)) if idx in lime_payload else None
            rows.append(
                {
                    "instance_id": str(idx),
                    "prediction_type": target_column,
                    "prediction_value": float(probability),
                    "predicted_class": str(pred_class),
                    "model_name": model_name,
                    "prediction_date": timestamp,
                    "probability": float(probability),
                    "shap_values": shap_json,
                    "lime_values": lime_json,
                }
            )
    else: # regression
        for idx, pred_value in zip(instance_ids, y_pred):
            shap_json = json.dumps(shap_payload.get(idx)) if idx in shap_payload else None
            lime_json = json.dumps(lime_payload.get(idx)) if idx in lime_payload else None
            rows.append(
                {
                    "instance_id": str(idx),
                    "prediction_type": target_column,
                    "prediction_value": float(pred_value),
                    "predicted_class": None,
                    "model_name": model_name,
                    "prediction_date": timestamp,
                    "probability": None,
                    "shap_values": shap_json,
                    "lime_values": lime_json,
                }
            )
    return pd.DataFrame(rows)


def _feature_importance(model: Any, feature_names: List[str]) -> Optional[List[float]]:
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_.tolist()
    return None


def train_models_and_explain(
    engine: Engine,
    feature_df: pd.DataFrame,
    target_column: str,
    task_type: str,
    selected_model: str,
    paths_config: PathsConfig | None = None,
    shap_sample_size: int = 200,
) -> ModelArtifact:
    """Train a selected model, generate explanations, and persist predictions."""
    paths = paths_config or get_paths_config()
    X_train, X_test, y_train, y_test, feature_names = prepare_datasets(feature_df, target_column, task_type)
    
    model = None
    if task_type == "classification":
        if selected_model == "RandomForestClassifier":
            model = _train_random_forest_classifier(X_train, y_train)
        elif selected_model == "GradientBoostingClassifier":
            model = _train_gradient_boosting_classifier(X_train, y_train)
        elif selected_model == "XGBClassifier":
            model = _train_xgboost_classifier(X_train, y_train)
        class_names = [str(c) for c in y_train.unique()]
    else: # regression
        if selected_model == "RandomForestRegressor":
            model = _train_random_forest_regressor(X_train, y_train)
        elif selected_model == "GradientBoostingRegressor":
            model = _train_gradient_boosting_regressor(X_train, y_train)
        elif selected_model == "XGBRegressor":
            model = _train_xgboost_regressor(X_train, y_train)
        class_names = None

    if model is None:
        raise ValueError(f"Unknown model: {selected_model}")

    sample = X_test.iloc[: min(shap_sample_size, len(X_test))]

    if task_type == "classification":
        metrics, y_pred, y_prob = _compute_classification_metrics(model, X_test, y_test)
    else:
        metrics, y_pred = _compute_regression_metrics(model, X_test, y_test)
        y_prob = None

    shap_values, expected_value = _generate_shap_values(model, X_train, sample)
    shap_array_2d = np.asarray(shap_values)
    if shap_array_2d.ndim == 1:
        shap_array_2d = shap_array_2d.reshape(1, -1)
    shap_payload = {
        idx: {feat: float(val) for feat, val in zip(feature_names, shap_row.flatten())}
        for idx, shap_row in zip(sample.index, shap_array_2d)
    }

    lime_payload = _generate_lime_explanations(model, X_train, sample, feature_names, task_type, class_names)
    lime_serializable = {
        int(idx): [{"feature": feat, "weight": float(weight)} for feat, weight in contribs]
        for idx, contribs in lime_payload.items()
    }

    predictions_df = _assemble_predictions_df(
        instance_ids=X_test.index,
        model_name=selected_model,
        target_column=target_column,
        task_type=task_type,
        y_pred=y_pred,
        shap_payload=shap_payload,
        lime_payload=lime_serializable,
        y_prob=y_prob,
    )

    predictions_df.to_sql(
        "prediction_results",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=500,
    )

    artifact = ModelArtifact(
        model_name=selected_model,
        model=model,
        metrics=metrics,
        feature_importance=_feature_importance(model, feature_names),
        shap_values=shap_array_2d,
        shap_expected_value=expected_value,
        shap_sample=sample,
        lime_explanations=lime_payload,
        predictions=predictions_df,
    )

    save_json(
        {
            "model_name": selected_model,
            "metrics": metrics,
        },
        paths.output_dir / f"{selected_model}_metrics.json",
    )
    save_json(
        {
            "model_name": selected_model,
            "shap_explanations": {
                int(idx): {feature: float(value) for feature, value in shap_dict.items()}
                for idx, shap_dict in shap_payload.items()
            },
            "lime_explanations": lime_serializable,
        },
        paths.explanations_dir / f"{selected_model}_explanations.json",
    )

    shap_sample_summary = json.dumps({k: shap_payload[k] for k in list(shap_payload.keys())[:2]}, indent=2)
    lime_sample_summary = json.dumps({k: lime_serializable[k] for k in list(lime_serializable.keys())[:2]}, indent=2)

    summary_content = f"""
    Model: {selected_model}
    Metrics:
    {json.dumps(metrics, indent=2)}

    SHAP Explanations (sample):
    {shap_sample_summary}

    LIME Explanations (sample):
    {lime_sample_summary}
    """
    summary = summarize_text(summary_content)
    save_summary(summary, f"{selected_model}_summary.txt")

    return artifact
