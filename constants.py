from pathlib import Path
IMG_DIR = Path("artifacts/data/img_data")
CATEGORIES = ["normal", "failure"]
CSV_PATH = Path("artifacts/data/tabular_data/heart_failure.csv")
RF_SCORE = Path("artifacts/models/rf/rf_score.json")
RF_MODEL = Path("artifacts/models/rf/rf_model.pkl")
CNN_SCORE = Path("artifacts/models/cnn/cnn_score.json")
CNN_MODEL = Path("artifacts/models/cnn/cnn_model.keras")