from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import numpy as np
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI Chest Cancer Detection API"}

@app.post("/train")
def train():
    os.system("dvc repro")  # Trigger model training
    return {"message": "Training started successfully"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    image_path = f"uploaded_{file.filename}"
    with open(image_path, "wb") as f:
        f.write(file.file.read())

    # Run prediction
    result = clApp.classifier.predict(image_path)
    
    return {"prediction": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
