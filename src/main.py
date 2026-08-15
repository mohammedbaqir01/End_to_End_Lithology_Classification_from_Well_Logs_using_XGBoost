from fastapi import FastAPI
import pandas as pd
from joblib import load
from pathlib import Path
from src.features.engineer_features import build_features
from pydantic import BaseModel
from typing import Optional

MODEL_PATH = Path("models/xgb_classifier.pkl")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")

model = load(MODEL_PATH)
label_encoder = load(LABEL_ENCODER_PATH)

app = FastAPI()


class Data(BaseModel):
    source_file: Optional[str] = None
    well_name: Optional[str] = None

    DEPT: float
    FORCE_2020_LITHOFACIES_CONFIDENCE: float
    FORCE_2020_LITHOFACIES_LITHOLOGY: Optional[int] = None

    CALI: float
    RDEP: float
    RMED: float
    DTC: float
    NPHI: float
    PEF: float
    GR: float
    RHOB: float
    DRHO: float

    DEPTH_MD: float
    X_LOC: float
    Y_LOC: float
    Z_LOC: float

    BS: float

    Lithology: Optional[int] = None

@app.get("/")
def root():
    return {"message": "Lithology Classification API"}


@app.post("/predict")
async def predict(data: Data):
    df = pd.DataFrame([data.model_dump()])

    df = build_features(df,df)

    df = df.drop(
        columns=[
            "Lithology",
            "well_name",
            "source_file",
            "FORCE_2020_LITHOFACIES_LITHOLOGY",
        ],
        errors="ignore",
        )
    prediction = model.predict(df)
    prediction_label = label_encoder.inverse_transform(prediction)

    return {
    "prediction": prediction_label.tolist(),
    "class_id": prediction.tolist()
    }
