
🔬 Thyroid Cancer Risk Prediction System

<p align="center">
  <b>End-to-End Machine Learning Clinical Decision Support System</b><br>
  Predicting Thyroid Cancer Risk with Optimized Recall & Real-Time Deployment
</p>


<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange" />
  <img src="https://img.shields.io/badge/Deployment-Streamlit-red" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>



⸻

🚀 Overview

This project presents a machine learning-based clinical decision support system designed to predict thyroid cancer risk (malignant vs benign) using patient clinical data.

The system is optimized for high recall (sensitivity) — ensuring minimal missed cancer cases — and deployed as an interactive web application using Streamlit.

⸻

🎯 Problem Statement

Early detection of thyroid cancer is crucial, but misdiagnosis (especially false negatives) can be life-threatening.

This project addresses:
	•	❌ Missed cancer diagnoses
	•	❌ Lack of accessible predictive tools
	•	❌ Need for real-time clinical assistance

⸻

🧠 Solution Approach
	•	Built an end-to-end ML pipeline
	•	Evaluated multiple models
	•	Performed hyperparameter tuning
	•	Optimized for Recall (Sensitivity)
	•	Deployed as a real-time web application

⸻

🏗️ Project Architecture

Data → Preprocessing → Model Training → Hyperparameter Tuning → Evaluation → Deployment (Streamlit)


⸻

🧪 Features Used

Category	Features
👤 Demographics	Age, Gender
⚠️ Risk Factors	Family History, Radiation Exposure
🧪 Clinical	TSH, T3, T4, Nodule Size
⚕️ Lifestyle	Smoking, Obesity, Diabetes, Iodine Deficiency


⸻

📊 Model Performance

Metric	Value
🔍 Recall (Sensitivity)	~71%
⚖️ F1 Score	~0.42
📈 ROC-AUC	~0.62

💡 The model prioritizes recall to reduce false negatives, which is critical in cancer detection.

⸻

🌐 Streamlit App Features
	•	🎛️ Interactive patient input interface
	•	📊 Real-time malignancy prediction
	•	🎯 Adjustable classification threshold
	•	📈 Confidence visualization (progress bar)
	•	🧠 Clinical-style UI layout

⸻

🖥️ Demo Preview

(Add screenshots here after uploading to GitHub)

📸 Example:
- Input form UI
- Prediction result display
- Confidence bar


⸻

📁 Project Structure

📁 thyroid-cancer-prediction/
│
├── app.py                         # Streamlit app
├── thyroid_cancer_ML_v2.ipynb     # Model training notebook
├── thyroid_model.pkl              # Trained model
├── feature_columns.pkl            # Feature order
├── best_model_name.pkl            # Model metadata
├── requirements.txt
└── README.md


⸻

⚙️ Installation

1. Clone the Repository

git clone https://github.com/your-username/thyroid-cancer-prediction.git
cd thyroid-cancer-prediction

2. Install Dependencies

pip install -r requirements.txt


⸻

▶️ Run the Application

streamlit run app.py

Then open:

http://localhost:8501


⸻

⚙️ Model Training (Optional)

jupyter notebook thyroid_cancer_ML_v2.ipynb


⸻

🧠 Key Design Decisions
	•	✔ Removed Ethnicity & Country → avoid bias & deployment mismatch
	•	✔ Used Pipelines → consistent preprocessing
	•	✔ Optimized for Recall → clinical safety priority
	•	✔ Feature alignment via reindex() → prevents runtime errors

⸻

⚠️ Limitations
	•	Moderate ROC-AUC indicates limited class separability
	•	No imaging features (e.g., ultrasound shape)
	•	Class imbalance impacts precision

⸻

🔮 Future Improvements
	•	SHAP explainability integration
	•	Image-based diagnosis using CNN
	•	SMOTE / advanced imbalance handling
	•	Cloud deployment (AWS / Streamlit Cloud)

⸻

⚠️ Disclaimer

This project is for educational and research purposes only and is not a substitute for professional medical diagnosis.

⸻

👨‍💻 Author

Arijit Bhattacharjee
Final Year ML Project

⸻

⭐ Support

If you found this project useful:
	•	⭐ Star the repository
	•	🍴 Fork and improve
	•	📢 Share with others

⸻

📌 Quick Summary

A complete ML pipeline + deployed application for thyroid cancer risk prediction, optimized for recall and built for real-world usability.
:::

⸻

🚀 What Makes This “Premium”
	•	✔ Badges (professional look)
	•	✔ Architecture section
	•	✔ Structured narrative
	•	✔ Research + engineering balance
	•	✔ Ready for recruiters / viva

