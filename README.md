# Personal-Loan-Classifier
# Personal Loan Acceptance Classifier
## Overview
This repository contains a Jupyter Notebook for a Personal Loan Acceptance Classifier. The project focuses on analyzing a dataset related to bank marketing campaigns to predict whether a client will subscribe to a personal loan. The dataset includes various features such as client demographics, financial status, and previous interactions with the bank.The project also includes a simple **web-based application** that runs locally on your computer.

## Dataset
The dataset used in this project is bank-full.csv, which contains information about bank clients and their responses to marketing campaigns. The dataset includes the following features:

- age: Age of the client

- job: Type of job

- marital: Marital status

- education: Education level

- default: Has credit in default?

- balance: Average yearly balance

- housing: Has housing loan?

- loan: Has personal loan?

- contact: Contact communication type

- day: Last contact day of the month

- month: Last contact month of the year

- duration: Last contact duration (in seconds)

- campaign: Number of contacts performed during this campaign

- pdays: Number of days since the client was last contacted

- previous: Number of contacts performed before this campaign

- poutcome: Outcome of the previous marketing campaign

- y: Has the client subscribed to a personal loan? (Target variable)

## Project Structure
The notebook is structured into the following sections:

- Importing Required Libraries: Loads necessary Python libraries for data analysis and machine learning.

- Data Loading and Initial Exploration: Loads the dataset and performs initial exploration to understand its structure and contents.

- Data Visualization: Includes visualizations to explore the distribution of the target variable and other features.

- Data Preprocessing: Handles missing values, encodes categorical variables, and scales numerical features.

- Model Training and Evaluation: Trains multiple machine learning models and evaluates their performance using metrics like accuracy and classification reports.

- Deploy Deployment: This repository includes an app.py file containing a Streamlit web application for interactive loan acceptance predictions. The app loads trained models and allows users to input features via a user-friendly interface to get real-time predictions.(Due large model size the Models Pickle file can't be uploaded)

## Key Findings
- The dataset is imbalanced, with most clients not subscribing to a personal loan.

- Various machine learning models are trained, including Logistic Regression, K-Nearest Neighbors, Decision Trees, Random Forests, and Gradient Boosting.

- Techniques like SMOTE and RandomUnderSampler are used to handle class imbalance.

- Dependencies
To run this notebook, you need the following Python libraries:

pandas

seaborn

matplotlib

scikit-learn

xgboost

imbalanced-learn
