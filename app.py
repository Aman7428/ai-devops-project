from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline("sentiment-analysis")

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "AI DevOps API Running"}

@app.post("/predict")
def predict(data: TextInput):
    result = classifier(data.text)
    return {
        "input": data.text,
        "prediction": result
    }