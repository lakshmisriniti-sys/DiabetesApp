🩺 Diabetes Prediction App

An AI-powered web application that predicts whether a person is likely to have diabetes based on medical attributes such as glucose, blood pressure, BMI, and age.
Built using Machine Learning, FastAPI (backend), and Streamlit (frontend).

🎯 Overview

Diabetes is one of the most common chronic diseases globally.
This project aims to assist in early detection by predicting diabetes risk using trained machine learning models.

Users can input their health details in a simple Streamlit UI and get instant predictions powered by three trained algorithms — Random Forest, Logistic Regression, and Support Vector Machine (SVM).

🚀 Features

🧠 Prediction using three ML models — Random Forest, Logistic Regression, and SVM

⚙️ Fast and scalable FastAPI backend

💻 Streamlit frontend for intuitive user input

📊 Real-time model performance comparison

🔒 Validates inputs for realistic health ranges

🧾 Clean, minimal user interface

🧰 Tech Stack
Layer	Tools Used
Frontend	Streamlit
Backend	FastAPI
Machine Learning	Scikit-learn, Pandas, NumPy
Language	Python 3.10+
Version Control	Git, GitHub
Deployment (Planned)	Streamlit Cloud + Render
📈 Model Performance
Model	Accuracy	ROC-AUC	Best Feature	Notes
Random Forest Classifier	0.7532 (75.32%)	0.8304	Glucose	Best performing model overall
Logistic Regression	0.7532 (75.32%)	0.8242	Glucose	Interpretable and consistent
SVM (Support Vector Machine)	0.7532 (75.32%)	0.8103	BMI	Balanced performance
🧩 Random Forest Detailed Metrics:
Precision (0): 0.81   Recall (0): 0.81
Precision (1): 0.65   Recall (1): 0.65
Accuracy: 75.32%
Confusion Matrix:
[[80 19]
 [19 36]]
ROC-AUC: 0.8305

Logistic Regression Detailed Metrics:
Precision (0): 0.80   Recall (0): 0.83
Precision (1): 0.67   Recall (1): 0.62
Accuracy: 75.32%
Confusion Matrix:
[[82 17]
 [21 34]]
ROC-AUC: 0.8242

SVM Detailed Metrics:
Precision (0): 0.79   Recall (0): 0.85
Precision (1): 0.68   Recall (1): 0.58
Accuracy: 75.32%
Confusion Matrix:
[[84 15]
 [23 32]]
ROC-AUC: 0.8104


✅ The application currently uses the Random Forest Classifier for prediction as it achieved the highest ROC-AUC (0.83) and most balanced recall.

🧪 Dataset Information

Dataset: PIMA Indians Diabetes Database

Source: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)

Total Records: 768 samples, 8 features

Target: Outcome (0 = Non-diabetic, 1 = Diabetic)

Features Used:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

⚙️ Project Structure
DiabetesApp/
│
├── backend/
│   ├── main.py                 # FastAPI backend app
│   ├── model.pkl               # Saved trained model
│
├── scripts/
│   ├── frontend_app.py         # Streamlit frontend app
│
├── notebooks/
│   ├── model_training.ipynb    # Model training and evaluation
│
├── requirements.txt
├── README.md                   # Project documentation
└── .gitignore

🧩 How It Works

User Input:
Users enter health parameters in the Streamlit form.

Backend Processing:
Streamlit sends the data to FastAPI via a POST request.

Model Prediction:
The backend loads the trained model (model.pkl) and predicts the outcome.

Result Display:
Streamlit displays whether the user is “Diabetic” or “Non-Diabetic”, along with prediction confidence.

🧰 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/DiabetesApp.git
cd DiabetesApp

2️⃣ Create a Virtual Environment
python -m venv venv


Activate it:

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Backend (FastAPI)
uvicorn backend.main:app --reload


Visit Swagger UI 👉 http://127.0.0.1:8000/docs

5️⃣ Run the Frontend (Streamlit)
streamlit run scripts/frontend_app.py
### 📷 Screenshots

**1️⃣ Streamlit Input Interface**
![App Screenshot](assets/screenshot_ui.png)

**2️⃣ Prediction Result**
![Prediction Result](assets/screenshot_result.png)

**3️⃣ FastAPI Swagger Docs**
![API Screenshot](assets/screenshot_api.png)
### 🌐 Live Demo

- 🔗 **Streamlit Frontend:** [https://diabetesapp-4fnvao7vlr453axebukzjj.streamlit.app/](https://diabetesapp-4fnvao7vlr453axebukzjj.streamlit.app/)

- 🔗 **FastAPI Backend (Render):** [https://diabetesapp-n440.onrender.com/docs](https://diabetesapp-n440.onrender.com/predict/docs)
🏆 Future Enhancements

📊 Add a graph comparing different model performances

💾 Store patient history and predictions in a database

📱 Mobile-friendly interface

☁️ Deploy the app for public access

🧬 Integrate real-time health monitoring features

👩‍💻 Developer

Lakshmi Sriniti
📍 Cambridge, UK
💼 GitHub: https://github.com/lakshmisriniti-sys

📧 Email: lakshmisriniti@gmail.com

⚖️ License

This project is licensed under the MIT License
.

💡 Acknowledgements

Dataset by NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)

Built as part of AI & Machine Learning practice project

Inspired by the goal to leverage AI for early disease prediction



