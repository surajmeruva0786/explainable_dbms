# Explainable AI-Powered DBMS

An intelligent, LLM-powered machine learning platform that automatically analyzes datasets, generates complete ML pipelines, and provides comprehensive explainability through SHAP and LIME visualizations. The system features a modern web interface, automated code generation, and detailed activity logging.

## 🌟 Project Overview

This project bridges the gap between powerful machine learning models and users who need transparent, interpretable insights. Unlike traditional ML platforms that provide only predictions, this system explains **why** predictions were made, making AI accessible and trustworthy for decision-makers.

**Key Innovation**: The platform uses Google's Gemini LLM to automatically:
- Analyze datasets and suggest optimal target columns
- Recommend appropriate ML models based on data characteristics
- Generate complete preprocessing and training pipelines as executable Python code
- Answer natural language questions about model predictions and performance

## 🏗️ Architecture

### Technology Stack

**Backend**:
- **FastAPI**: High-performance REST API server
- **Python 3.8+**: Core application logic
- **SQLAlchemy**: Database ORM for MySQL
- **Google Gemini AI**: LLM-powered analysis and code generation

**Machine Learning**:
- **scikit-learn**: Classical ML algorithms and preprocessing
- **XGBoost**: Gradient boosting models
- **SHAP**: Global and local feature importance
- **LIME**: Instance-level explanations
- **Featuretools**: Automated feature engineering

**Frontend**:
- **React**: Modern web interface (located in `src/explainable_dbms/xai_dbms_frontend`)
- **Interactive Visualizations**: SHAP plots, LIME explanations, feature importance charts

**Logging & Analytics**:
- **Google Firestore**: Cloud-based activity logging via REST API
- Tracks: analysis sessions, LLM calls, user queries, model training events

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Frontend (React)                    │
│              Upload CSV → View Results → Ask Questions       │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST API
┌────────────────────────▼────────────────────────────────────┐
│                    FastAPI Backend (app.py)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ /api/upload  │  │ /api/analyze │  │  /api/query  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│  File Storage   │ │  LLM Pipeline    │ │  Query Handler  │
│  (temp_data/)   │ │  Generator       │ │  (LLM-powered)  │
└─────────────────┘ └────────┬─────────┘ └─────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Code Executor   │
                    │  (Safe Sandbox)  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │   MySQL DB  │  │  Artifacts  │  │  Firestore  │
    │ (Predictions)│  │  (Plots)    │  │  (Logs)     │
    └─────────────┘  └─────────────┘  └─────────────┘
```

## ✨ Key Features

### 1. **Intelligent Dataset Analysis**
- **LLM Column Extractor** (`llm_column_extractor.py`): Analyzes uploaded datasets and suggests optimal target columns for prediction
- Provides reasoning for suggestions based on data characteristics
- Fallback heuristics when LLM is unavailable

### 2. **Automated ML Pipeline Generation**
The system uses a three-stage LLM workflow:

1. **Analysis Advisor** (`llm_analysis_advisor.py`): 
   - Examines dataset structure and statistics
   - Recommends model type (classification/regression)
   - Suggests appropriate algorithms

2. **Pipeline Advisor** (`llm_pipeline_advisor.py`):
   - Designs preprocessing strategy (handling missing values, encoding, scaling)
   - Determines feature engineering approaches
   - Plans train/test split strategy

3. **Code Generator** (`llm_code_generator.py`):
   - Generates complete, executable Python code for the entire pipeline
   - Includes data loading, preprocessing, model training, and visualization
   - Creates SHAP and LIME explanations automatically

### 3. **Safe Code Execution**
- **Code Executor** (`code_executor.py`): Runs LLM-generated code in a controlled environment
- Captures stdout/stderr for debugging
- Timeout protection (300s default)
- Automatic artifact detection (plots, metrics)

### 4. **Explainable AI (XAI)**
- **SHAP Explanations**: 
  - Summary plots showing global feature importance
  - Force plots for individual predictions
  - Waterfall charts for instance-level analysis
  
- **LIME Explanations**:
  - Local interpretable model approximations
  - Top contributing features for each prediction
  - Visualizations for first 3 test instances

### 5. **Natural Language Query Interface**
- **Query Handler** (`query_handler.py`): LLM-powered Q&A system
- Ask questions like:
  - "What features are most important for this model?"
  - "Why was instance 42 classified as high risk?"
  - "How accurate is the model?"
- Answers grounded in actual analysis results (metrics, visualizations, data)

### 6. **Comprehensive Logging**
- **Firestore Logger** (`firestore_logger.py`): Asynchronous cloud logging
- Tracks:
  - Analysis start/completion events
  - LLM code generation attempts
  - User queries and answers
  - Model performance metrics
- Enables usage analytics and debugging

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- **MySQL Server** (for storing predictions and explanations)
- **Google Gemini API Key** (for LLM features)
- **Node.js** (for frontend development, optional)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/surajmeruva0786/explainable_dbms.git
   cd explainable_dbms
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```env
   # MySQL Database Configuration
   MYSQL_USER=your_mysql_user
   MYSQL_PASSWORD=your_mysql_password
   MYSQL_HOST=localhost
   MYSQL_PORT=3306
   MYSQL_DATABASE=explainable_dbms

   # Google Gemini API Key (required for LLM features)
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Build frontend** (optional, if modifying UI):
   ```bash
   cd src/explainable_dbms/xai_dbms_frontend
   npm install
   npm run build
   cd ../../..
   ```

### Running the Application

**Start the server**:
```bash
python -m src.explainable_dbms.app
```

The application will:
- Start FastAPI server on `http://127.0.0.1:8000`
- Automatically open your browser to the web interface
- Serve the React frontend and API endpoints

### Using the Web Interface

1. **Upload Dataset**: 
   - Click "Upload CSV" and select your dataset file
   - System analyzes columns and suggests targets

2. **Configure Analysis**:
   - Select target column (or let LLM suggest one)
   - Click "Analyze" to start ML pipeline

3. **View Results**:
   - Model performance metrics
   - SHAP summary plot (global feature importance)
   - LIME explanations (first 3 test instances)
   - Confusion matrix / regression plots

4. **Ask Questions**:
   - Use natural language to query results
   - Examples:
     - "What is the model accuracy?"
     - "Which features are most important?"
     - "Explain the prediction for instance 5"

## 📁 Project Structure

```
explainable_dbms/
├── .env                          # Environment configuration
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
│
├── artifacts/                    # Generated plots and analysis results
│   └── <analysis_id>/            # Per-session artifacts
│       ├── shap_summary.png
│       ├── lime_instance_*.png
│       ├── metrics.json
│       └── state.pkl             # Session state for queries
│
├── temp_data/                    # Uploaded CSV files
├── outputs/                      # Legacy output directory
│
└── src/explainable_dbms/
    ├── __init__.py
    │
    ├── app.py                    # FastAPI server & API endpoints
    ├── config.py                 # Application configuration
    ├── models.py                 # SQLAlchemy database models
    │
    ├── part1_setup.py            # Database initialization
    ├── part2_schema_data.py      # Data loading & feature engineering
    ├── part3_ml_xai.py           # Model training & XAI generation
    ├── part4_visualization.py    # Plot generation (SHAP/LIME)
    ├── part5_evaluation.py       # Model evaluation utilities
    │
    ├── llm_column_extractor.py   # LLM-based target column suggestion
    ├── llm_analysis_advisor.py   # Dataset analysis & model recommendation
    ├── llm_pipeline_advisor.py   # Preprocessing strategy generation
    ├── llm_code_generator.py     # Complete pipeline code generation
    ├── llm_model_selector.py     # Model selection logic
    ├── llm_summarizer.py         # Text summarization
    │
    ├── code_executor.py          # Safe execution of generated code
    ├── query_handler.py          # Natural language Q&A
    ├── firestore_logger.py       # Cloud logging to Firestore
    ├── io_utils.py               # File I/O utilities
    │
    └── xai_dbms_frontend/        # React web application
        ├── public/
        ├── src/
        ├── build/                # Production build (served by FastAPI)
        └── package.json
```

## 🔧 API Endpoints

### `POST /api/upload`
Upload a CSV dataset file.

**Request**: Multipart form data with CSV file  
**Response**:
```json
{
  "filename": "dataset.csv",
  "message": "File uploaded successfully"
}
```

### `POST /api/analyze`
Trigger ML analysis with LLM-generated pipeline.

**Request**:
```json
{
  "filename": "dataset.csv",
  "target_column": "price"  // Optional, LLM will suggest if empty
}
```

**Response**:
```json
{
  "message": "Analysis complete",
  "analysis_id": "uuid-here",
  "plots": {
    "shap_summary": "/artifacts/uuid/shap_summary.png",
    "lime_instance_0": "/artifacts/uuid/lime_instance_0.png"
  },
  "model": "XGBRegressor",
  "target": "price",
  "output": "Training logs..."
}
```

### `POST /api/query`
Ask questions about completed analysis.

**Request**:
```json
{
  "query": "What is the model accuracy?",
  "analysis_id": "uuid-here"
}
```

**Response**:
```json
{
  "answer": "The model achieved 94.2% accuracy on the test set...",
  "plot_url": null
}
```

## 🧪 Machine Learning Models

The system supports both **classification** and **regression** tasks:

### Classification Models
- **RandomForestClassifier**: Ensemble of 300 decision trees with balanced class weights
- **GradientBoostingClassifier**: Boosted trees with learning rate 0.05
- **XGBClassifier**: Optimized gradient boosting with regularization

### Regression Models
- **RandomForestRegressor**: Ensemble regressor with 300 estimators
- **GradientBoostingRegressor**: Boosted regression trees
- **XGBRegressor**: High-performance gradient boosting

**Model Selection**: LLM analyzes dataset characteristics and recommends the most appropriate model based on:
- Target variable distribution
- Dataset size and dimensionality
- Feature types (numeric/categorical)
- Business context

## 📊 Explainability Methods

### SHAP (SHapley Additive exPlanations)
- **Theory**: Game-theoretic approach to explain model outputs
- **Global Explanations**: Summary plots showing average feature impact
- **Local Explanations**: Force plots for individual predictions
- **Advantages**: Consistent, theoretically sound, model-agnostic

### LIME (Local Interpretable Model-agnostic Explanations)
- **Theory**: Approximates complex models locally with interpretable models
- **Instance Explanations**: Shows top contributing features for specific predictions
- **Advantages**: Intuitive, works with any model, highlights local decision boundaries

## 🔐 Security & Safety

- **Code Validation**: Generated code is checked for dangerous operations
- **Sandboxed Execution**: LLM-generated code runs in controlled environment
- **Timeout Protection**: 300-second execution limit prevents infinite loops
- **File Access Control**: Code restricted to `temp_data/` and `artifacts/` directories

## 🌐 Cloud Integration

### Firestore Logging
All user activities are logged to Google Firestore for analytics:

**Collections**:
- `user_activity`: Analysis start events
- `analyses`: Completed analysis sessions with metrics
- `llm_calls`: Code generation attempts and results
- `queries`: User questions and LLM answers

**Configuration**: Update `firestore_logger.py` with your Firebase project credentials.

## 🛠️ Development

### Testing LLM Components

```bash
# Test Gemini API connection
python test_api_key.py

# Test available models
python test_available_models.py

# List all Gemini models
python list_models.py

# Test all model types
python test_all_models.py
```

### Database Management

The system uses MySQL to store:
- **Predictions**: Model outputs with probabilities
- **Explanations**: SHAP and LIME values (JSON)
- **Metadata**: Timestamps, model names, instance IDs

Tables are created automatically on first run via `part1_setup.py`.

## 📝 Example Workflow

1. **Upload** `customer_churn.csv` with columns: `age`, `income`, `usage_minutes`, `churned`
2. **LLM Analysis**: System suggests `churned` as target (binary classification)
3. **Model Recommendation**: XGBClassifier selected based on dataset size and class imbalance
4. **Pipeline Generation**: LLM creates code for:
   - Handling missing values (mean imputation)
   - Encoding categorical features
   - Train/test split (75/25)
   - Model training with hyperparameters
   - SHAP/LIME explanation generation
5. **Execution**: Code runs safely, generates visualizations
6. **Results**: 
   - Accuracy: 92.3%
   - Top features: `usage_minutes`, `income`, `age`
   - SHAP plot shows global importance
   - LIME explains why customer #42 churned
7. **Query**: "Why did customer 42 churn?"
   - **Answer**: "Customer 42 was predicted to churn primarily due to low usage_minutes (15 mins/month) and declining income trend..."

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional ML models (neural networks, ensemble methods)
- More XAI techniques (Anchors, Counterfactuals)
- Enhanced frontend visualizations
- Support for time-series and text data
- Multi-model comparison dashboards

## 📄 License

This project is open-source. Please check the repository for license details.

## 🙏 Acknowledgments

- **Google Gemini AI**: Powers intelligent analysis and code generation
- **SHAP Library**: Provides robust explainability framework
- **LIME**: Enables local interpretable explanations
- **FastAPI**: High-performance async web framework
- **React**: Modern frontend library

## 📧 Contact

For questions or support, please open an issue on the GitHub repository.

---

**Built with ❤️ to make AI transparent and trustworthy**