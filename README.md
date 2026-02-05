# AI-engineering-bootcamp
My learning path to ai engineering 

## Month 1 : Foundation of Data Engineering and Machine Learning
- **project:**

### Titanic Survival Predictor (84.9% Accuracy)

#### 🚢 The Mission

This project goes beyond training a single machine learning model.  
The goal was to build a **complete end-to-end ML pipeline**, from data preprocessing and feature engineering to model evaluation and explainability.

The focus was on:
- Improving model performance through meaningful features
- Comparing different modeling approaches
- Making predictions interpretable rather than treating the model as a black box

---

### 🔧 Key Engineering Highlights

### Feature Engineering
- Created **FamilySize** by combining `SibSp` and `Parch`
- Derived **IsAlone** from `FamilySize`
- These features improved model accuracy from **~78% to 84.9%**

### Model Comparison
- Trained and evaluated multiple models
- Compared **Random Forest** performance against a **Neural Network**
- Selected the final model based on validation accuracy and stability

### Explainability
- Used **feature importance** to understand which variables influenced predictions
- Helped validate model behavior and reduce black-box risk

---

### 🧰 Tech Stack

- Python  
- Pandas  
- Scikit-Learn  
- Keras / TensorFlow  
- Streamlit  
- Joblib  

---

### ▶️ How to Run the App

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Run the Streamlit app:
   ```bash
    streamlit run app.py
    

