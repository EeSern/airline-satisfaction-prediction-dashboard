# Airline Passenger Satisfaction Prediction

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end data analytics solution developed for the **Introduction to Data Analytics (AICT009-4-2-IDA)** course at **Asia Pacific University of Technology & Innovation (APU)**. This project analyzes passenger survey data to build, evaluate, and deploy a high-performance XGBoost model for predicting customer satisfaction.

The final deliverable is a fully functional, interactive web application built with Streamlit, which provides real-time predictions, batch processing, and a human-in-the-loop feedback system.

---

## 🏛️ Project Context & Team

This repository represents the final submission for a group assignment. The project was developed collaboratively by:
- **Heng Ee Sern** 
- **Tan Hao Shuan** 
- **Laeu Zi-Li** 

---

## ✨ Key Features

-   **Comprehensive EDA:** In-depth exploratory data analysis to uncover initial patterns and insights.
-   **Rigorous Feature Selection:** A multi-stage process to distill 23 initial features down to an optimal set of 10.
-   **Comparative Modeling:** Head-to-head evaluation of Logistic Regression, Random Forest, and XGBoost.
-   **High-Performance Champion Model:** A fine-tuned **XGBoost** classifier achieving **94.25% accuracy** on unseen data.
-   **Explainable AI (XAI):** Model predictions are interpreted using **SHAP**, providing transparency into the "black box" model's decision-making process.
-   **Interactive Streamlit Dashboard:** A live application with three functional tabs for analysis, single prediction, and batch prediction.

---

## 📂 Repository Structure
```
├── .gitignore
├── LICENSE
├── README.md
├── notebook/
│   ├── IDA_Assignment_Team_2.ipynb         # Main analysis notebook
│   ├── airline_passenger_satisfaction.csv  # Raw dataset used by the notebook
│   └── data_dictionary.csv                 # Dataset feature descriptions
├── report/
│   ├── IDA Group Final Report Team 2.pdf     # Final written report
│   └── IDA Proposal Team 2.pdf               # Initial project proposal
└── streamlit_app/
    ├── app.py                              # Main Streamlit application script
    ├── requirements.txt                    # Python dependencies for the app
    ├── final_xgboost_model.joblib          # The trained and exported model
    ├── final_features.json                 # The 10 features the model requires
    ├── airline_data_final_10_features.csv  # Template file for batch prediction
    ├── airline_data_for_charts.csv         # Cleaned data for EDA charts
    └── ...                                 # Other data files generated/used by the app
```

---

## 🔬 1. The Colab Analysis Notebook (`notebook/IDA_Assignment_Team_2.ipynb`)

This Jupyter Notebook contains the full end-to-end data science workflow, including:
-   Data loading, cleaning, and preprocessing.
-   Extensive Exploratory Data Analysis (EDA).
-   A comparative analysis of multiple feature selection methods (Filter, Wrapper, Embedded).
-   A head-to-head comparison of three machine learning models.
-   Fine-tuning of the champion model (XGBoost) using GridSearchCV.
-   Advanced model interpretation using SHAP.
-   Exporting of the final model and assets for deployment.

### How to Run the Notebook
The notebook is designed to be run in a **Google Colab** environment. It requires the raw dataset files (`airline_passenger_satisfaction.csv`, `data_dictionary.csv`) to be placed in a user's Google Drive and the path to be updated in the notebook accordingly.

---

## 🚀 2. The Streamlit Dashboard (`streamlit_app/`)

This folder contains the final deployed application, an interactive dashboard that allows users to explore the data and get real-time predictions.

### How to Run the Dashboard Locally

**Prerequisites:**
-   Python 3.9+
-   `pip` and `venv` (recommended)

**Setup Instructions:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/EeSern/airline-satisfaction-prediction-dashboard.git
    cd airline-satisfaction-prediction-dashboard
    ```

2.  **Navigate to the Application Directory:**
    ```bash
    cd streamlit_app
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate on Windows
    venv\Scripts\activate
    
    # Activate on macOS/Linux
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    Install all the required Python libraries using the `requirements.txt` file located in this directory.
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Application:**
    Launch the Streamlit server with the following command.
    ```bash
    streamlit run app.py
    ```

The application should now be running and accessible in your local web browser.
