# ❤️ Heart Disease Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts the presence of heart disease using patient-related clinical features. The project explores multiple classification algorithms, performs hyperparameter optimization, evaluates model performance, and provides model explainability through SHAP.

The best-performing model, XGBoost, is integrated into a Streamlit application for interactive predictions.

## 🚀 Project Overview

This project demonstrates a complete machine learning workflow, from data exploration and preprocessing to model training, evaluation, explainability, and deployment.

**Key Highlights**
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature transformation
- Comparison of multiple classification algorithms
- Hyperparameter tuning using RandomizedSearchCV
- Stratified 5-fold cross-validation
- Model evaluation using multiple performance metrics
- Feature importance using Permutation Importance
- Model explainability using SHAP
- Interactive prediction interface using Streamlit

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit
- Jupyter Notebook

## 🤖 Machine Learning Algorithms

The project evaluates the following classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. Support Vector Machine (SVM)
6. Gradient Boosting
7. Extra Trees Classifier
8. XGBoost Classifier

Hyperparameter tuning is performed on selected models using RandomizedSearchCV.

## 📊 Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

### Best Model: XGBoost

| Metric | Result |
|---|---:|
| Cross-Validation ROC-AUC | 70.60% |
| Test ROC-AUC | 71.40% |

The final model is selected based on cross-validation performance and subsequently evaluated on an unseen test dataset.

## 🔍 Model Explainability

To improve model interpretability, the project incorporates:

- **Permutation Feature Importance:** Identifies features that contribute to predictive performance.
- **SHAP Analysis:** Explains global feature contributions and individual model predictions.

## 🌐 Streamlit Application

The project includes an interactive Streamlit application that provides:

- Patient clinical feature inputs
- Heart disease prediction
- Predicted probability visualization
- Interactive user interface

## 📁 Project Structure

```text
Heart-Disease-Prediction-ML/
│
├── Heart_Disease_End_to_End_ML_GitHub.ipynb
├── app.py
├── README.md
├── requirements.txt
├── heart_disease_raw_practice.csv
│
├── saved_models/
│   └── heart_disease_xgboost.pkl
│
└── assets/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Heart-Disease-Prediction-ML.git
```

### 2. Navigate to the project directory

```bash
cd Heart-Disease-Prediction-ML
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```
### 5. Run the Streamlit application
```bash
streamlit run app.py
```
## 📌 Future Enhancements

- Improve model performance through additional experimentation.
- Add interactive model performance visualizations.
- Enhance prediction explanations with SHAP.
- Deploy the Streamlit application online.

## ⚠️ Disclaimer

This project is intended strictly for educational and demonstration purposes. It uses a synthetic practice dataset and is not validated for real-world medical diagnosis or clinical decision-making.

└── artifacts/
```
