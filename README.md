# Explainable DBMS Pipeline

This project implements an end-to-end explainable analytical database pipeline that connects MySQL with Python-based analytics, machine learning, and explainability tooling (SHAP/LIME), culminating in interactive dashboards and evaluation reports.

## Project Structure

```
src/explainable_dbms/
├─ config.py                # Environment-aware configuration helpers
├─ io_utils.py              # Shared I/O helpers for JSON/CSV artifacts
├─ models.py                # SQLAlchemy ORM models
├─ part1_setup.py           # Database creation and schema management
├─ part2_schema_data.py     # Synthetic data generation and feature engineering
├─ part3_ml_xai.py          # Model training, prediction storage, SHAP/LIME
├─ part4_visualization.py   # Plotly/Matplotlib dashboards
├─ part5_evaluation.py      # Performance + explanation quality evaluation
└─ run_pipeline.py          # Orchestrates the full workflow
```

Artifacts are written to `outputs/`:

- `outputs/visualizations/` for HTML/PNG charts and dashboards
- `outputs/explanations/` reserved for explanation JSON exports
- `outputs/benchmark_results.json` for evaluation metrics
- Snapshot CSV extracts for quick inspection (`customers_snapshot.csv`, etc.)

## Installation

1. Ensure you have a running MySQL instance and credentials with permissions to create databases and tables.
2. Set environment variables as needed (defaults in parentheses):
   - `MYSQL_USER` (`root`)
   - `MYSQL_PASSWORD` (`password`)
   - `MYSQL_HOST` (`127.0.0.1`)
   - `MYSQL_PORT` (`3306`)
   - `MYSQL_DATABASE` (`explainable_dbms`)
   - `PROJECT_ROOT` (defaults to current working directory)
3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the full pipeline:

```bash
python -m explainable_dbms.run_pipeline
```

Each stage can also be executed independently by importing and invoking the relevant module functions.

### Outputs

- Predictions and explanations are persisted to the MySQL `prediction_results` table.
- SHAP/LIME graphics are saved to `outputs/visualizations/` in both HTML and PNG formats.
- Benchmark metrics (latency, explanation quality, simulated user uplift) are saved to `outputs/benchmark_results.json`.

## Reproducibility Checklist

- Seeded random number generators ensure deterministic synthetic datasets.
- Model training uses fixed random states.
- The pipeline re-creates tables with the latest generated data upon each run.

## Notes

- The evaluation module includes a simulated user study aligned with the expected improvements specified in the prompt.
- Adjust hyperparameters, sample sizes, and evaluation settings in the respective modules to suit production workloads.

