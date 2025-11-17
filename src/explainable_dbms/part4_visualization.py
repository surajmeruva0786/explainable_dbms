"""
Part 4: Visualization of SHAP and LIME explanations.
"""

from __future__ import annotations

from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

from .part3_ml_xai import ModelArtifact


def shap_summary_plot(artifact: ModelArtifact) -> go.Figure:
    """Create SHAP summary plot for the supplied model."""
    shap_values = np.abs(artifact.shap_values).mean(axis=0)
    feature_names = list(artifact.shap_sample.columns)
    order = np.argsort(shap_values)[::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=shap_values[order],
                y=[feature_names[i] for i in order],
                orientation="h",
            )
        ]
    )
    fig.update_layout(
        title=f"SHAP Summary Plot - {artifact.model_name}",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
    )
    return fig


def shap_waterfall_plot(artifact: ModelArtifact) -> go.Figure:
    """Create SHAP waterfall plot for the first sample instance."""
    fig = go.Figure()
    if artifact.shap_values.shape[0] > 0:
        base_value = artifact.shap_expected_value
        if isinstance(base_value, list):
            base_value = base_value[0]

        shap_values_for_instance = artifact.shap_values
        
        if shap_values_for_instance.ndim > 1:
            shap_values_for_instance = shap_values_for_instance[0]

        explanation = shap.Explanation(
            values=shap_values_for_instance,
            base_values=base_value,
            data=artifact.shap_sample.iloc[0].values,
            feature_names=artifact.shap_sample.columns.tolist(),
        )
        
        fig = go.Figure(go.Waterfall(
            name = "Prediction", orientation = "h",
            measure = ["relative"] * len(explanation.feature_names),
            y = explanation.feature_names,
            x = explanation.values,
            connector = {"mode":"between", "line":{"width":4, "color":"rgb(0, 0, 0)", "dash":"solid"}}
        ))
        fig.update_layout(
            title=f"SHAP Waterfall Plot - {artifact.model_name}",
            showlegend = True
        )
    return fig


def shap_bar_plot(artifact: ModelArtifact) -> go.Figure:
    """Bar chart of mean absolute SHAP values."""
    shap_values = np.abs(artifact.shap_values).mean(axis=0)
    feature_names = list(artifact.shap_sample.columns)
    order = np.argsort(shap_values)[::-1]

    fig = go.Figure(
        data=[
            go.Bar(
                x=shap_values[order],
                y=[feature_names[i] for i in order],
                orientation="h",
            )
        ]
    )
    fig.update_layout(
        title=f"SHAP Mean Absolute Contributions - {artifact.model_name}",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
    )
    return fig


def lime_local_plot(artifact: ModelArtifact) -> go.Figure:
    """Visualise LIME explanation for the first sample instance."""
    fig = go.Figure()
    if artifact.lime_explanations:
        first_idx = next(iter(artifact.lime_explanations))
        contributions = artifact.lime_explanations[first_idx]
        features, weights = zip(*contributions)
        fig.add_trace(
            go.Bar(x=weights, y=features, orientation="h"),
        )
        fig.update_layout(
            title=f"LIME Local Explanation - {artifact.model_name} (customer {first_idx})",
            xaxis_title="Contribution weight",
            yaxis_title="Feature",
        )
    return fig


def combined_dashboard(artifact: ModelArtifact) -> go.Figure:
    """Create a combined dashboard comparing SHAP and LIME outputs."""
    shap_values = np.abs(artifact.shap_values).mean(axis=0)
    feature_names = list(artifact.shap_sample.columns)
    order = np.argsort(shap_values)[::-1]
    
    first_idx = next(iter(artifact.lime_explanations), None)
    lime_features, lime_weights = ([], [])
    if first_idx is not None:
        lime_features, lime_weights = zip(*artifact.lime_explanations[first_idx])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("SHAP Global Importance", "LIME Local Importance"),
    )
    fig.add_trace(
        go.Bar(
            x=shap_values[order],
            y=[feature_names[i] for i in order],
            orientation="h",
            name="SHAP",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=lime_weights,
            y=lime_features,
            orientation="h",
            name="LIME",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=f"SHAP vs LIME Comparison Dashboard - {artifact.model_name}",
        height=500,
        width=1100,
        showlegend=False,
    )
    return fig


def generate_visualizations(artifacts: List[ModelArtifact]) -> Dict[str, Dict[str, go.Figure]]:
    """Create all required visualisations for each model."""
    visualizations = {}
    for artifact in artifacts:
        visualizations[artifact.model_name] = {
            "summary": shap_summary_plot(artifact),
            "waterfall": shap_waterfall_plot(artifact),
            "bar": shap_bar_plot(artifact),
            "lime": lime_local_plot(artifact),
            "comparison": combined_dashboard(artifact),
        }
    return visualizations
