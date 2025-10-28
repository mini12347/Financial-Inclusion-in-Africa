# 🌍 Financial Inclusion Prediction App

### 🧠 Predicting Bank Account Ownership in Africa using Machine Learning & Streamlit
# Dataset source : Zindi Platform
---

## 📘 Overview
The **Financial Inclusion Prediction App** is a machine learning web application built to **predict whether a person owns a bank account** based on various demographic and socioeconomic factors such as **country, location type, gender, cellphone access, education level**, and more.

This project aims to highlight and analyze the **financial inclusion gap in Africa**, helping policymakers and organizations understand key barriers to banking access.

---

## 🚀 Features
- 🤖 **Machine Learning Model (XGBoost):** Classifies individuals as likely or unlikely to have a bank account.  
- 💻 **Interactive Streamlit Web Interface:** Intuitive UI where users can input details and get instant predictions.  
- 📊 **Dashboard Page:** Displays interactive charts showing trends and relationships between financial access and demographics.  
- 🧩 **Data Analysis Notebook:** Includes exploratory data analysis, preprocessing, and model training in Jupyter Notebook.  
- ☁️ **Deployed Online:** Accessible from any browser via Streamlit Cloud.

---

## ⚙️ Technologies Used

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.12 |
| **Libraries** | pandas, numpy,ydata_profiling, plotly, seaborn, matplotlib, scikit-learn, xgboost, joblib, streamlit |
| **Tools & Platforms** | Jupyter Notebook, VS Code, Git, GitHub, Streamlit Cloud |

---

## 📂 Project Structure
📁 Financial-Inclusion-Prediction <br/>
├── Financial_inclusion_dataset.csv # Dataset .csv <br/>
├── Notebook.ipynb # Model training, analysis, and evaluation <br/>
├── report.ipynb # ydata_profiling report <br/>
├── Stream.py # Streamlit web app <br/>
├── xgb_model.pkl # Saved XGBoost model <br/>
├── requirements.txt # Project dependencies <br/>
└── README.md # Documentation



---

## 🧭 Data Description
The dataset focuses on **financial inclusion in African countries** and includes columns such as:

| Feature | Description |
|----------|-------------|
| `country` | Country of the individual |
| `year` | Year of the survey |
| `location_type` | Urban or Rural |
| `cellphone_access` | Whether the person has cellphone access |
| `household_size` | Number of people in the household |
| `gender_of_respondent` | Male or Female |
| `education_level` | Highest level of education attained |
| `bank_account` | Target variable: Yes (1) or No (0) |

---

## 🧩 Model Development Workflow

1. **Data Cleaning & Preprocessing**  
   - Handled missing values, inconsistent labels, and categorical encoding.  
   - Normalized and prepared data for model training.

2. **Exploratory Data Analysis (EDA)**  
   - Used Plotly and Seaborn to visualize trends in banking inclusion.  
   - Discovered correlations between cellphone access, education, and financial inclusion.

3. **Model Training (XGBoost Classifier)**  
   - Tuned hyperparameters with GridSearchCV.  
   - Addressed class imbalance with class weights and evaluation metrics (F1-score, ROC AUC).

4. **Model Exporting**  
   - Saved trained model as `model.pkl` using `joblib`.

5. **Streamlit Integration**  
   - Designed an intuitive interface for user input and live predictions.  
   - Added a dashboard page with interactive charts.

---

## 💻 Installation & Setup

### 1️⃣ Clone the Repository
git clone https://github.com/yourusername/financial-inclusion-prediction.git
cd financial-inclusion-prediction
2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Locally
streamlit run Stream.py
