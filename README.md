🚀 End-to-End Chest Cancer Detection using MLOps & DVC

🌟 An AI-powered solution for early chest cancer detection using Deep Learning and MLOps.

📌 Integrated with DVC for dataset versioning & automated CI/CD workflows.

🔬 Deployed via Streamlit & FastAPI for real-time predictions.

Dataset : https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

🏥 Project Overview
Chest cancer detection from medical images is a crucial application of AI in healthcare. This project follows a complete Machine Learning (ML) Lifecycle using MLOps principles and Data Version Control (DVC) to ensure:

✅ Scalability - Efficient dataset management with DVC.

✅ Reproducibility - Automated pipelines for training & deployment.

✅ Automation - Continuous Integration (CI) & Deployment (CD) via GitHub Actions.

✅ Real-Time Predictions - Deployable via Streamlit & FastAPI web apps.


🖥 Live Demo: Chest Cancer Detection App

📂 Project Structure
bash
Copy
Edit
📦 end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc

│-- .dvc/               # DVC configuration for dataset & model versioning

│-- .github/workflows/   # CI/CD automation with GitHub Actions

│-- config/             # Configuration files for training & inference

│-- research/           # Jupyter notebooks for exploratory data analysis

│-- src/cnnClassifier/  # CNN architecture & ML pipeline scripts

│-- templates/          # HTML templates for the web app interface

│-- app.py              # Flask-based web API

│-- streamlit_app.py    # Streamlit app for user-friendly predictions

│-- FastAPI_app.py      # FastAPI backend for RESTful predictions

│-- requirements.txt    # Dependencies for setting up the project

🛠️ Installation & Setup
1️⃣ Clone the Repository

bash

Copy

Edit

git clone https://github.com/aimlproject083-rgb/End_to_End_Chest_Cancer_Detection_ML_Project_using_DVC_and_MLflow.git

cd end-to-end-ml_project-chest-cancer-detection-using-mlops-and-dvc

2️⃣ Create a Virtual Environment
bash
Copy
Edit
python3 -m venv chest
source chest/bin/activate  # For Windows use: chest\Scripts\activate
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Install DVC & Fetch Data
bash
Copy
Edit
pip install dvc
dvc pull  # Pull dataset & model files
🎯 Model Training & Evaluation
🔵 Configure Training Parameters
Modify the config/config.yaml file with the dataset path and hyperparameters.

🔵 Start Training
bash
Copy
Edit
python src/cnnClassifier/train.py
The model will be trained and saved for deployment.

🔵 Evaluate Performance
bash
Copy
Edit
python src/cnnClassifier/evaluate.py
This will generate accuracy, loss curves, and confusion matrices.

🚀 Deployment Options
🟢 1. Streamlit Web App
Run the Streamlit app for an interactive user interface:

bash
Copy
Edit
streamlit run streamlit_app.py
📌 Features:
✔ Upload X-ray images
✔ Get real-time predictions
✔ Confidence score displayed

🖥 Live Demo: [Chest Cancer Detection App](https://endtoendchestcancerdetectionaimlproject-6gcnrho.streamlit.app/)

🔵 2. FastAPI REST API
Run the FastAPI-based backend for model inference:

bash
Copy
Edit
uvicorn FastAPI_app:app --reload
📌 Access API Docs: http://127.0.0.1:8000/docs
✔ Upload images as JSON payload
✔ Get predictions via RESTful API

🔥 MLOps & DVC Integration
🚀 MLOps Workflows:

✅ GitHub Actions: Automates testing, training, and deployment.

✅ DVC (Data Version Control): Keeps track of datasets & models.

✅ Pipeline Orchestration: Ensures smooth training-to-deployment transition.

💡 Why DVC?

Reproducibility: Always use the correct dataset version.

Collaboration: Work seamlessly across multiple systems.

Efficiency: Fetch only required data, saving storage & bandwidth.

bash
Copy
Edit
# Track dataset
dvc add data/dataset
git commit -m "Tracked dataset using DVC"
git push origin main

# Pull dataset in a new environment
dvc pull
💡 Results & Performance
📌 The model has been trained on chest X-ray images to classify:

Normal

Adenocarcinoma (Left Lower Lobe - Stage Ib)

Large Cell Carcinoma (Left Hilum - Stage IIIa)

Squamous Cell Carcinoma (Left Hilum - Stage IIIa)


🤝 Contributing
💡 Contributions, issues, and feature requests are welcome!
Feel free to check the issues page.

📜 License
This project is licensed under the MIT License.

📬 Contact

📌 Author: Aryan Dhanuka , Aryan Upadhayay and Kushal Bansal









