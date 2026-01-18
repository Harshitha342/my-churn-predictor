# Customer Churn Prediction – End-to-End ML Pipeline

## 📌 Overview
This project implements an end-to-end **machine learning pipeline** to predict **customer churn** for a telecommunications company.  
It covers **data preprocessing**, **model training and evaluation**, **REST API deployment**, **containerization with Docker**, and **automated testing**.

The goal is to demonstrate practical skills in **applied machine learning** and **MLOps** using Python.

---

## 🧠 Problem Statement
Customer churn occurs when customers stop using a company’s services. Predicting churn enables businesses to proactively retain customers through targeted actions.

This system predicts whether a customer will churn (`Yes` / `No`) along with the **probability of churn**.

---

## 🗂 Dataset
- **Name:** Telco Customer Churn Dataset (IBM)
- **Source:** Kaggle (official mirror)
- **Rows:** 7,043  
- **Features:** 20  
- **Target:** `Churn` (Yes / No)

> Dataset was sourced from Kaggle due to deprecated public links, while maintaining identical structure and columns.

---

## ⚙️ Tech Stack
- **Python 3**
- **pandas, numpy**
- **scikit-learn**
- **Flask**
- **joblib**
- **pytest**
- **Docker**

---

## 🏗 Project Structure

my-churn-predictor/
├── app/
│ ├── init.py
│ ├── main.py # Flask API
│ ├── model.py # Model training & preprocessing
│ └── utils.py
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│ └── churn_model.joblib
├── tests/
│ ├── test_api.py
│ ├── test_model.py
│ └── conftest.py
├── Dockerfile
├── requirements.txt
├── README.md
└── .env.example

---

## 🔬 Model Details
- **Algorithm:** Logistic Regression
- **Preprocessing:**
  - OneHotEncoding for categorical features
  - StandardScaler for numerical features
  - Handling missing values
- **Train/Test Split:** 80/20 (stratified)

### 📊 Evaluation Metrics (on Test Set)
- **Accuracy:** ~80%
- **Precision:** ~65%
- **Recall:** ~55%
- **F1-score:** ~60%

> Precision, Recall, and F1-score are emphasized due to class imbalance in churn prediction.

---

## 🚀 Running the Application (Local)

### 1️⃣ Setup Virtual Environment

python -m venv .venv
source .venv/Scripts/activate   # Git Bash
### 2️⃣ Install Dependencies
pip install -r requirements.txt

###  3️⃣ Train the Model
python app/model.py

### 4️⃣ Start the API
python app/main.py


API runs at:
http://127.0.0.1:8000

Available endpoint:
POST /predict


🔌 API Usage
Endpoint
POST /predict

Request Body (JSON)
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 900.0
}

Response
{
  "prediction": "Yes",
  "probability": 0.67
}

Error Handling

400 Bad Request → missing/invalid input

500 Internal Server Error → server issues

🧪 Running Tests
pytest


✔ Includes:

model existence & prediction tests

API success and validation tests

🐳 Running with Docker
Build Image
docker build -t churn-predictor .

Run Container
docker run -p 8000:8000 churn-predictor


API will be accessible at:

http://127.0.0.1:8000

🧩 Design Decisions

Logistic Regression chosen for interpretability and fast inference

scikit-learn Pipeline used to prevent data leakage

Single saved pipeline ensures preprocessing consistency

Dockerized deployment ensures reproducibility

📌 Conclusion

This project demonstrates a production-ready ML workflow from raw data to deployment.
It reflects real-world practices in ML engineering and MLOps, including testing, API development, and containerization.

👩‍💻 Author

Harshitha Arlapalli