#  Water Potability Prediction System

A Streamlit-based Machine Learning Web Application that predicts whether water is safe for drinking based on its physical and chemical properties.

##  Live Demo


## 🔗 Live Demo
 [Water Potability Prediction App](https://water-portability-prediction-project-fafbhfzaffxmlvyyfdha9d.streamlit.app)

---

##  Project Overview

Access to safe drinking water is one of the most important public health concerns worldwide. This project leverages Machine Learning algorithms to analyze water quality parameters and predict whether the water is potable (safe for drinking) or non-potable (unsafe for drinking).

The application provides real-time predictions through an interactive Streamlit interface and compares the performance of multiple classification models.

---

##  Objectives

* Predict whether water is safe for drinking.
* Compare multiple machine learning classification algorithms.
* Improve model performance using hyperparameter tuning.
* Visualize evaluation metrics and confusion matrices.
* Provide an easy-to-use web interface for end users.

---

##  Features

 Predicts water potability instantly

 Interactive Streamlit user interface

Real-time machine learning predictions

Comparison of multiple ML algorithms

  Hyperparameter tuning using GridSearchCV

 Confusion Matrix visualization

 Precision, Recall, and F1-Score analysis

 Model performance comparison

---

##  Technology Stack

| Category                | Technologies        |
| ----------------------- | ------------------- |
| Programming Language    | Python              |
| Data Processing         | Pandas, NumPy       |
| Machine Learning        | Scikit-Learn        |
| Visualization           | Matplotlib, Seaborn |
| Web Framework           | Streamlit           |
| Model Serialization     | Pickle              |
| Development Environment | Jupyter Notebook    |

---

##Project Structure

```text
water-potability-predictor/
│
├── app.py
├── model/
│   ├── water_model.pkl
│   └── scaler.pkl
│
├── data/
│   └── water_potability.csv
│
├── notebooks/
│   └── model_training.ipynb
│
├── requirements.txt
├── README.md
└── screenshots/
```

##  Dataset Information

Dataset Source: Kaggle Water Potability Dataset

### Dataset Details

* Total Records: 3,276
* Features: 9
* Target Variable: Potability

### Input Features

1. pH
2. Hardness
3. Solids
4. Chloramines
5. Sulfate
6. Conductivity
7. Organic Carbon
8. Trihalomethanes
9. Turbidity

### Target Variable

| Value | Meaning                       |
| ----- | ----------------------------- |
| 1     | Potable (Safe to Drink)       |
| 0     | Not Potable (Unsafe to Drink) |

---

##  Machine Learning Models Used

### Logistic Regression

* Baseline Classification Model
* Fast and Interpretable

### Decision Tree Classifier

* Rule-Based Classification
* Easy Visualization

### Support Vector Machine (SVM)

* Strong Generalization Performance
* Effective for Binary Classification

### K-Nearest Neighbors (KNN)

* Instance-Based Learning
* Good Classification Accuracy

### Naive Bayes

* Probabilistic Classification
* Fast Training and Prediction

---

## 📈 Model Performance

| Algorithm           | Tuning Status     | Training Accuracy | Testing Accuracy |
| ------------------- | ----------------- | ----------------- | ---------------- |
| Logistic Regression | Without Tuning    | 0.60              | 0.62             |
| Logistic Regression | With GridSearchCV | 0.60              | 0.62             |
| Decision Tree       | Without Tuning    | 1.00              | 0.59             |
| Decision Tree       | With GridSearchCV | 0.67              | 0.64             |
| SVM                 | Without Tuning    | 0.73              | 0.69             |
| SVM                 | With GridSearchCV | 0.73              | 0.69             |
| KNN                 | Without Tuning    | 0.75              | 0.63             |
| KNN                 | With GridSearchCV | 1.00              | 0.65             |

---

##  Evaluation Metrics

The models were evaluated using:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix

Confusion matrices were visualized using Seaborn heatmaps for better model interpretation.

---

##  Hyperparameter Tuning

GridSearchCV was used to optimize model parameters and improve prediction performance.

Benefits:

* Improved Generalization
* Reduced Overfitting
* Better Model Selection

---

## 💡 Key Findings

* SVM achieved the most balanced performance.
* Hyperparameter tuning improved Decision Tree and KNN performance.
* KNN showed strong training accuracy after tuning.
* SVM maintained good precision and recall on unseen data.

---

## Installation & Setup

### Clone Repository

```bash
git clone https://github.com/your-username/water-potability-predictor.git
```

### Navigate to Project

```bash
cd water-potability-predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

##  Requirements

```text
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
pickle-mixin
```

---

## 🌱 Future Enhancements

* Random Forest Classifier
* XGBoost Integration
* Deep Learning Models
* SHAP Explainability
* Docker Deployment
* Cloud-Based Model Hosting
* Automated Data Updates

---

##  Learning Outcomes

Through this project, I gained practical experience in:

* Data Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Scaling
* Classification Algorithms
* Hyperparameter Tuning
* Model Evaluation
* Streamlit Deployment
* End-to-End Machine Learning Workflow

---

## Author

**Pasupuleti Mounika**

B.Tech – Computer Science Engineering (AI & ML)

PVP Siddhartha Institute of Technology

📧 Email: [mounikapossibility72@gmail.com](mailto:mounikapossibility72@gmail.com)

---

##  Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

Your support helps motivate further development and improvements.



