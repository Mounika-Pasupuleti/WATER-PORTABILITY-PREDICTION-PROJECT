


# 💧 Water Potability Prediction App

A Streamlit-based machine learning web app that predicts whether water is **safe to drink** based on various physical and chemical features.

---

## 📌 Overview

This project uses classification algorithms like **Logistic Regression, Decision Tree, SVM, KNN, and Naive Bayes** to predict the **potability of water**. The aim is to assist in early detection of unfit water sources and promote safe drinking practices.

---

## 🚀 Features

* 🧪 Predicts **potable** or **non-potable** water
* 📊 Real-time predictions using trained ML models
* 📈 Displays model accuracies with and without tuning
* ✅ Shows **confusion matrix**, **precision**, **recall**, **F1-score**
* 💡 Compares model performance (with GridSearchCV)

---

## 🧰 Tech Stack

| Category      | Tools & Libraries   |
| ------------- | ------------------- |
| Programming   | Python              |
| Data Handling | Pandas, NumPy       |
| ML Models     | Scikit-learn        |
| Visualization | Matplotlib, Seaborn |
| Deployment    | Streamlit           |
| Model Saving  | Pickle              |

---

## 📂 Project Structure

```
water-potability-predictor/
│
├── app.py                  # Streamlit web app
├── model/                  
│   ├── water_model.pkl     # Trained ML model
│   └── scaler.pkl          # Feature scaler
├── data/
│   └── water_potability.csv
├── notebooks/
│   └── model_training.ipynb  # Model training + EDA
├── requirements.txt
└── README.md
```

---

## 📈 Dataset Information

* **Source**: [Kaggle – Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
* **Records**: 3,276 samples
* **Target**: `Potability` (1 = safe, 0 = unsafe)
* **Features**:

  * pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/water-potability-predictor.git
cd water-potability-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## ✅ Model Performance Summary

| Algorithm           | Tuning Status     | Training Acc | Testing Acc |
| ------------------- | ----------------- | ------------ | ----------- |
| Logistic Regression | Without Tuning    | 0.60         | 0.62        |
| Logistic Regression | With GridSearchCV | 0.60         | 0.62        |
| Decision Tree       | Without Tuning    | 1.00         | 0.59        |
| Decision Tree       | With GridSearchCV | 0.67         | 0.64        |
| SVM                 | Without Tuning    | 0.73         | 0.69        |
| SVM                 | With GridSearchCV | 0.73         | 0.69        |
| KNN                 | Without Tuning    | 0.75         | 0.63        |
| KNN                 | With GridSearchCV | 1.00         | 0.65        |

---

## 📊 Confusion Matrix Visualization

Each algorithm includes a visual confusion matrix heatmap using `Seaborn`.

---

## 📝 Conclusion

* Among all models, **SVM and KNN with tuning** provided **better generalization** on unseen data.
* **Hyperparameter tuning** improved performance, especially in Decision Tree and KNN.
* **SVM** provided a strong balance between precision and recall.

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Save this in your `requirements.txt`.

---

## 💡 Future Scope

* Add advanced models like **Random Forest**, **XGBoost**
* Implement **model explainability** (e.g., SHAP values)
* Deploy using **Docker** for scalable access

---

## 🙋‍♀️ Author

**Pasupuleti Mounika**
🎓 B.Tech CSE (AI & ML), PVP Siddhartha Institute of Technology
🔗 [LinkedIn – Possibility Mounika](https://www.linkedin.com/in/possibilitymounika)
📧 [mounikapossibility72@gmail.com](mailto:mounikapossibility72@gmail.com)


### 🌟 If you liked this project, give it a ⭐ and share it with others!


