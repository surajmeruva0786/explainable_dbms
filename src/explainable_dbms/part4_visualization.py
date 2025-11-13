"""
Part 4: Visualization of SHAP and LIME explanations.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap

from .config import PathsConfig, get_paths_config
from .io_utils import save_json
from .part3_ml_xai import ModelArtifact


def _save_matplotlib(fig: plt.Figure, html_path: Path, png_path: Path) -> None:
    """Persist Matplotlib figures as PNG; export HTML using Plotly conversion if needed."""
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    with png_path.open("rb") as fh:
        encoded = base64.b64encode(fh.read()).decode("utf-8")
    html_content = f"""
    <html>
      <head><title>{png_path.stem}</title></head>
      <body style="margin:0;display:flex;justify-content:center;background:#111;">
        <img src="data:image/png;base64,{encoded}" alt="{png_path.stem}" style="max-width:90vw;max-height:90vh;"/>
      </body>
    </html>
    """
    html_path.write_text(html_content, encoding="utf-8")


def shap_summary_plot(artifact: ModelArtifact, paths: PathsConfig) -> None:
    """Create SHAP summary plot for the supplied model."""
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        artifact.shap_values,
        artifact.shap_sample,
        feature_names=list(artifact.shap_sample.columns),
        show=False,
    )
    plt.title(f"SHAP Summary Plot - {artifact.model_name}")
    html_path = paths.visualizations_dir / f"{artifact.model_name}_shap_summary.html"
    png_path = paths.visualizations_dir / f"{artifact.model_name}_shap_summary.png"
    _save_matplotlib(fig, html_path, png_path)


def shap_waterfall_plot(artifact: ModelArtifact, paths: PathsConfig) -> None:
    """Create SHAP waterfall plot for the first sample instance."""
    if artifact.shap_values.shape[0] == 0:
        return

    # Extract base_value as a scalar
    base_value_raw = artifact.shap_expected_value
    if hasattr(base_value_raw, '__len__') and not isinstance(base_value_raw, str):
        base_value = float(np.asarray(base_value_raw).flatten()[0])
    else:
        base_value = float(base_value_raw)
    
    # Ensure we get a single 1D array for the waterfall plot
    sample_row = artifact.shap_sample.iloc[0]
    feature_names_list = list(sample_row.index)
    n_features = len(feature_names_list)
    
    shap_vals_raw = artifact.shap_values[0]
    if hasattr(shap_vals_raw, 'flatten'):
        shap_vals_flat = shap_vals_raw.flatten()
    elif hasattr(shap_vals_raw, 'tolist'):
        shap_vals_flat = np.array(shap_vals_raw.tolist()).flatten()
    else:
        shap_vals_flat = np.array(shap_vals_raw).flatten()
    
    # Ensure shap_vals matches the number of features
    if len(shap_vals_flat) > n_features:
        shap_vals = shap_vals_flat[:n_features]
    elif len(shap_vals_flat) < n_features:
        # Pad with zeros if needed
        shap_vals = np.pad(shap_vals_flat, (0, n_features - len(shap_vals_flat)), mode='constant')
    else:
        shap_vals = shap_vals_flat
    
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=sample_row.values[:n_features] if len(sample_row.values) > n_features else sample_row.values,
        feature_names=feature_names_list,
    )
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    plt.title(f"SHAP Waterfall Plot - {artifact.model_name}")
    html_path = paths.visualizations_dir / f"{artifact.model_name}_shap_waterfall.html"
    png_path = paths.visualizations_dir / f"{artifact.model_name}_shap_waterfall.png"
    _save_matplotlib(fig, html_path, png_path)


def shap_bar_plot(artifact: ModelArtifact, paths: PathsConfig) -> None:
    """Bar chart of mean absolute SHAP values."""
    # Flatten to get mean absolute SHAP values for each feature
    shap_flat = artifact.shap_values.flatten()
    # Reshape to match feature count if needed
    n_features = len(artifact.shap_sample.columns)
    if len(shap_flat) > n_features:
        shap_flat = shap_flat[:n_features]
    
    mean_abs = np.abs(shap_flat)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=0)
    feature_names = list(artifact.shap_sample.columns)
    order = np.argsort(mean_abs)[::-1].tolist()

    fig = go.Figure(
        data=[
            go.Bar(
                x=mean_abs[order],
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
    html_path = paths.visualizations_dir / f"{artifact.model_name}_shap_bar.html"
    png_path = paths.visualizations_dir / f"{artifact.model_name}_shap_bar.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    fig.write_image(str(png_path))


def lime_local_plot(artifact: ModelArtifact, paths: PathsConfig) -> None:
    """Visualise LIME explanation for the first sample instance."""
    if not artifact.lime_explanations:
        return
    first_idx = next(iter(artifact.lime_explanations))
    contributions = artifact.lime_explanations[first_idx]
    features, weights = zip(*contributions)
    fig = go.Figure(
        data=[
            go.Bar(x=weights, y=features, orientation="h"),
        ]
    )
    fig.update_layout(
        title=f"LIME Local Explanation - {artifact.model_name} (customer {first_idx})",
        xaxis_title="Contribution weight",
        yaxis_title="Feature",
    )
    html_path = paths.visualizations_dir / f"{artifact.model_name}_lime_local.html"
    png_path = paths.visualizations_dir / f"{artifact.model_name}_lime_local.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    fig.write_image(str(png_path))


def combined_dashboard(artifact: ModelArtifact, paths: PathsConfig) -> None:
    """Create a combined dashboard comparing SHAP and LIME outputs."""
    # Flatten to get mean absolute SHAP values for each feature
    shap_flat = artifact.shap_values.flatten()
    # Reshape to match feature count if needed
    n_features = len(artifact.shap_sample.columns)
    if len(shap_flat) > n_features:
        shap_flat = shap_flat[:n_features]
    
    mean_abs = np.abs(shap_flat)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=0)
    feature_names = list(artifact.shap_sample.columns)
    order = np.argsort(mean_abs)[::-1].tolist()
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
            x=mean_abs[order],
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
    html_path = paths.visualizations_dir / f"{artifact.model_name}_shap_vs_lime.html"
    png_path = paths.visualizations_dir / f"{artifact.model_name}_shap_vs_lime.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    fig.write_image(str(png_path))


def generate_visualizations(artifacts: List[ModelArtifact], paths_config: PathsConfig | None = None) -> None:
    """Create all required visualisations for each model."""
    paths = paths_config or get_paths_config()
    manifest = {}
    for artifact in artifacts:
        shap_summary_plot(artifact, paths)
        shap_waterfall_plot(artifact, paths)
        shap_bar_plot(artifact, paths)
        lime_local_plot(artifact, paths)
        combined_dashboard(artifact, paths)
        manifest[artifact.model_name] = {
            "summary": f"{artifact.model_name}_shap_summary.html",
            "waterfall": f"{artifact.model_name}_shap_waterfall.html",
            "bar": f"{artifact.model_name}_shap_bar.html",
            "lime": f"{artifact.model_name}_lime_local.html",
            "comparison": f"{artifact.model_name}_shap_vs_lime.html",
        }
    save_json(manifest, paths.visualizations_dir / "visualization_manifest.json")

