# German Credit-Risk Classification Model
A machine learning model that predicts credit risk for loan applicants using the German Credit Risk dataset. Built with python using mainly scikit-learn, this project demonstrates the complete ML workflow from EDA to model deployment.

## Overview
This project aims to help financial institutions assess loan applicants' credit worthiness by predicting whether a borrower will default (bad risk) or repay (good risk). The model analyzes demographic and financial characteristics to provide risk assessments that can better inform lending decisions.

Key Capabilities:
- Predicts default probability for loan applicants
- Identifies key risk factors influencing credit decisions
- Provides an interactive web interface for real-time predictions

## Dataset
The german-credit-risk dataset contains 1,000 loan applications with 10 features:
- "Age":	Applicant's age
- "Sex":	Gender (male/female)
- "Job":	Employment category (0-3)
- "Housing":	Housing status (free/rent/own)
- "Saving accounts":	Savings account balance category
- "Checking account":	Checking account balance category
- "Credit amount":	Loan amount requested
- "Duration":	Loan term in months
- "Purpose":	Loan purpose
- "Risk":	(Target variable) good/bad credit risk

## Project Workflow
### 1. Data Import & Exploration
Loaded and inspected dataset structure. Identified missing values (18.3% in Saving accounts, 39.4% in Checking account). Analyzed distributions of numerical and categorical features.

### 2. Data Preprocessing
Handled missing values in "Saving accounts" and "Checking account" by treating them as "no account" (key assumption). Applied binary encoding for binary variables. Implemented ordinal encoding for categorical variables based on risk levels.

### 3. Exploratory Data Analysis
Visualized feature distributions and relationships. Analyzed default rates across variables. Identified purpose-based risk patterns.

### 4. Model Development
Trained baseline models (Random Forest & XGBoost) - Random Forest baseline was better. Performed hyperparameter tuning using RandomizedSearchCV.Optimized for recall to catch defaults ("bad" risks).

### 5. Evaluation
Achieved 76.3% accuracy with tuned Random Forest. 93% recall for good credit customers. 37% recall for bad credit customers (best of all models). 76.3% overal accuracy.

### 6. Deployment
Exported model using joblib. Created interactive Gradio interface. Implemented input validation and user-friendly interface.

## Key Insights
- Credit amount was the strongest predictor - Larger loans ("Credit amount") correlated with higher default risk.
- Banking relationships matter - Customers without checking accounts showed higher risk.
- Purpose drives risk - Business "Purpose" loans were safest (least defaults); vacation loans were riskiest (most defaults).
- Gender has minimal impact - "Sex" is the least important feature

## Limitations
- Dataset was imbalanced (70% "good", 30% "bad" risks)
- Limited sample size (1,000 client records only)
- Simplified missing value treatment during imputation.

# 👤 Author
## Brenton Mutetwa

GitHub: github.com/brentmutetwa

LinkedIn:zw.linkedin.com/in/brenton-mutetwa-95470b253  

Email: mutetwabrentont@gmail.com

