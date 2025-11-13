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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from sqlalchemy.engine import Engine

from .config import PathsConfig, get_paths_config
from .io_utils import save_json


FEATURE_COLUMNS = [
    "age",
    "income",
    "credit_score",
    "transaction_count",
    "total_spending",
    "avg_transaction_amount",
    "unique_categories",
    "avg_discount",
    "days_since_last_transaction",
]


@dataclass
class ModelArtifact:
    """Container for fitted models and explainability metadata."""

    model_name: str
    model: Any
    accuracy: float
    classification_report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    feature_importance: Optional[List[float]]
    shap_values: np.ndarray
    shap_expected_value: Any
    shap_sample: pd.DataFrame
    lime_explanations: Dict[int, List[Tuple[str, float]]]
    predictions: pd.DataFrame


def prepare_datasets(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the aggregated feature dataframe into train/test sets."""
    dataset = feature_df.set_index("customer_id")
    X = dataset[FEATURE_COLUMNS]
    y = dataset["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


def _train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
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


def _train_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=400,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
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
) -> Dict[int, List[Tuple[str, float]]]:
    """Generate LIME explanations with top contributing features."""
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_names,
        class_names=["retain", "churn"],
        mode="classification",
        discretize_continuous=True,
        verbose=False,
    )

    explanations: Dict[int, List[Tuple[str, float]]] = {}
    for idx, row in X_sample.iterrows():
        exp = explainer.explain_instance(
            data_row=row.values,
            predict_fn=model.predict_proba,
            num_features=min(5, len(feature_names)),
        )
        explanations[idx] = exp.as_list()
    return explanations


def _compute_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[float, Dict[str, Any], List[List[int]], np.ndarray, np.ndarray]:
    """Generate standard classification metrics and predictions."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cmatrix = confusion_matrix(y_test, y_pred).tolist()
    return acc, report, cmatrix, y_pred, y_prob


def _assemble_predictions_df(
    customer_ids: pd.Index,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    shap_payload: Dict[int, Any],
    lime_payload: Dict[int, Any],
) -> pd.DataFrame:
    """Create a dataframe ready for persistence in the prediction_results table."""
    rows = []
    timestamp = datetime.utcnow()
    for idx, probability, pred_class in zip(customer_ids, y_prob, y_pred):
        shap_json = json.dumps(shap_payload.get(idx)) if idx in shap_payload else None
        lime_json = json.dumps(lime_payload.get(idx)) if idx in lime_payload else None
        rows.append(
            {
                "customer_id": int(idx),
                "prediction_type": "customer_churn",
                "prediction_value": float(probability),
                "predicted_class": "churn" if pred_class == 1 else "retain",
                "model_name": model_name,
                "prediction_date": timestamp,
                "probability": float(probability),
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
    paths_config: PathsConfig | None = None,
    shap_sample_size: int = 200,
) -> List[ModelArtifact]:
    """Train models, generate explanations, and persist predictions."""
    paths = paths_config or get_paths_config()
    X_train, X_test, y_train, y_test = prepare_datasets(feature_df)
    feature_names = list(X_train.columns)

    models = {
        "RandomForestClassifier": _train_random_forest(X_train, y_train),
        "GradientBoostingClassifier": _train_gradient_boosting(X_train, y_train),
        "XGBClassifier": _train_xgboost(X_train, y_train),
    }

    artifacts: List[ModelArtifact] = []
    sample = X_test.iloc[: min(shap_sample_size, len(X_test))]

    for model_name, model in models.items():
        accuracy, report, cmatrix, y_pred, y_prob = _compute_metrics(model, X_test, y_test)

        shap_values, expected_value = _generate_shap_values(model, X_train, sample)
        shap_payload = {
            idx: dict(zip(feature_names, shap_row))
            for idx, shap_row in zip(sample.index, shap_array)
        }

        lime_payload = _generate_lime_explanations(model, X_train, sample, feature_names)
        lime_serializable = {
            int(idx): [{"feature": feat, "weight": float(weight)} for feat, weight in contribs]
            for idx, contribs in lime_payload.items()
        }

        predictions_df = _assemble_predictions_df(
            customer_ids=X_test.index,
            y_prob=y_prob,
            y_pred=y_pred,
            model_name=model_name,
            shap_payload=shap_payload,
            lime_payload=lime_serializable,
        )

        predictions_df.to_sql(
            "prediction_results",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=500,
        )

        artifact = ModelArtifact(
            model_name=model_name,
            model=model,
            accuracy=accuracy,
            classification_report=report,
            confusion_matrix=cmatrix,
            feature_importance=_feature_importance(model, feature_names),
            shap_values=shap_array,
            shap_expected_value=expected_value,
            shap_sample=sample,
            lime_explanations=lime_payload,
            predictions=predictions_df,
        )
        artifacts.append(artifact)

        save_json(
            {
                "model_name": model_name,
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": cmatrix,
            },
            paths.output_dir / f"{model_name}_metrics.json",
        )
        save_json(
            {
                "model_name": model_name,
                "shap_explanations": {
                    int(idx): {feature: float(value) for feature, value in shap_dict.items()}
                    for idx, shap_dict in shap_payload.items()
                },
                "lime_explanations": lime_serializable,
            },
            paths.explanations_dir / f"{model_name}_explanations.json",
        )

    return artifacts

