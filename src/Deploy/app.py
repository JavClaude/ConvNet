from io import BytesIO
import json
from typing import List

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import uvicorn

from src.Model.baseline import ConvNet


device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(file) -> List[float]:
    image = Image.open(BytesIO(file))
    preprocess_image = preprocessor(image)
    _, preds = model(preprocess_image)
    return preds.cpu().tolist()

with open("config_file.json", "r") as file:
    config_file = json.load(file)

preprocessor = Compose(
           [
                ToTensor(),
                Normalize((0.1307,), (0.3081,))
           ]
        )

model = ConvNet(**config_file)
model.load_state_dict(torch.load("model.pt", map_location=device))

app = FastAPI()

@app.post("/model/score")
async def predict_endpoint(file: UploadFile = File(...)):
    image = await file.read()
    pred = predict(image)
    return {"prediction": pred}


if __name__ == "__main__":
    
    uvicorn.run(app, port=8080)