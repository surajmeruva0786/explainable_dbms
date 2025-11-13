from setuptools import setup, find_packages

setup(
    name="explainable_dbms",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "mysql-connector-python",
        "sqlalchemy",
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "shap",
        "lime",
        "matplotlib",
        "seaborn",
        "plotly",
        "kaleido",
        "scipy",
    ],
    python_requires=">=3.8",
)

