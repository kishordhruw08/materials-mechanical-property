from fastapi import FastAPI
from pydantic import BaseModel
from models.predict import predict_properties

app = FastAPI(title="Materials Property Predictor")

class MaterialInput(BaseModel):
    formula: str


@app.post("/predict")
def predict(input: MaterialInput):
    return predict_properties(input.formula)
