# 🚢 Titanic Survival Prediction - Data Analysis Pipeline

> A machine learning pipeline designed to analyze passenger data and predict survival probabilities based on socio-economic status, age, and gender.

## 📖 Project Overview
This project demonstrates a standard **Data Science Workflow**, focusing on **Data Quality** and **Feature Engineering**. It serves as a practice ground for automated data validation and model testing.

## 🛠️ Tech Stack
- **Language:** Python 3.9
- **Libraries:** Pandas (Data Manipulation), Scikit-learn (Machine Learning)
- **Data Source:** [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)

## 📊 Key Features & Logic
1.  **Data Cleaning Pipeline:**
    -   Automated handling of missing values in `Age` and `Embarked` columns.
    -   Categorical data encoding for `Sex` (Male/Female mapping).
2.  **Model Training:**
    -   Utilizes **Random Forest Classifier** for robust prediction.
    -   Splits data into Training/Testing sets (80/20) to validate accuracy.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone https://github.com/Ra1nForest/Titanic-Data-Analysis.git
    ```
2.  Install dependencies:
    ```bash
    pip install pandas scikit-learn
    ```
3.  Run the analysis script:
    ```bash
    python analysis.py
    ```

---
*Created by [Qinglin Lu] - UNSW Master of IT Student*
