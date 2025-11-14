# Explainable AI-Powered DBMS

This project provides an interactive command-line application that allows users to upload their own datasets, train machine learning models, and receive easy-to-understand explanations for the model's predictions. It leverages the power of Explainable AI (XAI) to demystify the inner workings of complex models, making them more transparent and trustworthy.

## Project Overview

The core idea behind this project is to bridge the gap between powerful but often opaque machine learning models and the users who need to make decisions based on their outputs. Instead of just providing a prediction, our application explains *why* a certain prediction was made.

For example, if a model predicts that a customer is likely to churn, the application can highlight the key factors that contributed to this prediction, such as the customer's recent purchase history, their income level, or their credit score.

## Workflow

The application follows a simple, step-by-step workflow:

1.  **Upload Dataset**: The user starts by providing a path to their dataset in CSV format.
2.  **Select Target Column**: The user specifies which column in the dataset they want to predict (the "target column").
3.  **Model Training**: The application automatically trains a suite of machine learning models (Random Forest, Gradient Boosting, and XGBoost) on the provided dataset to predict the target column.
4.  **XAI Explanation Generation**: For each model, the application uses SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to generate detailed explanations for the model's predictions.
5.  **Interactive Q&A**: After the models are trained, the user can ask questions in plain English about specific data points. For example, a user can ask: "Why was customer 123 predicted to churn?"
6.  **Get Explanations**: The application will then provide a detailed explanation, including:
    *   A natural language summary of the prediction.
    *   A SHAP force plot, which is a visualization that shows the factors that pushed the prediction higher or lower.

## Technical Aspects

### Backend

*   **Language**: Python 3
*   **Database**: MySQL (for storing predictions and explanations)
*   **Machine Learning**:
    *   `scikit-learn`: For building and training machine learning models.
    *   `xgboost`: For the high-performance XGBoost model.
*   **Explainable AI (XAI)**:
    *   `shap`: For generating SHAP explanations.
    *   `lime`: For generating LIME explanations.
*   **LLM Summarizer**: A custom module that uses a Large Language Model to generate natural language summaries of the explanations.

### Machine Learning Models

The application trains the following three models:

*   **Random Forest**: An ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
*   **Gradient Boosting**: A machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees.
*   **XGBoost**: An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

### Explainable AI (XAI)

*   **SHAP (SHapley Additive exPlanations)**: A game theoretic approach to explain the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.
*   **LIME (Local Interpretable Model-agnostic Explanations)**: An algorithm that can explain the predictions of any classifier or regressor in a faithful way, by approximating it locally with an interpretable model.

## How to Use

### Prerequisites

*   Python 3.8 or higher
*   A running MySQL instance

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/explainable_dbms.git
    cd explainable_dbms
    ```

2.  **Set up environment variables**:
    Create a `.env` file in the root of the project and add the following variables:

    ```
    MYSQL_USER=your_mysql_user
    MYSQL_PASSWORD=your_mysql_password
    MYSQL_HOST=your_mysql_host
    MYSQL_PORT=your_mysql_port
    MYSQL_DATABASE=explainable_dbms
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the main pipeline**:
    ```bash
    python -m src.explainable_dbms.run_pipeline
    ```

2.  **Follow the prompts**:
    *   The application will first ask for the path to your CSV dataset.
    *   Then, it will ask for the name of the column you want to predict.

3.  **Ask questions**:
    Once the models are trained, you can ask questions about your data, for example:
    ```
    Ask a question about your data (or type 'exit' to quit): why did customer 123 churn?
    ```

## File Structure

```
.
├── .env                  # Environment variables for database connection
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── setup.py              # Project setup script
└── src
    └── explainable_dbms
        ├── __init__.py
        ├── config.py             # Configuration for the application
        ├── io_utils.py           # Utility functions for input/output
        ├── llm_summarizer.py     # Module for generating natural language summaries
        ├── models.py             # SQLAlchemy models for the database
        ├── part1_setup.py        # Database setup and initialization
        ├── part2_schema_data.py  # Data loading and feature engineering
        ├── part3_ml_xai.py       # Machine learning and XAI logic
        ├── part4_visualization.py# Visualization generation
        └── run_pipeline.py       # Main entry point of the application
```