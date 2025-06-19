from fastapi import FastAPI
from pydantic import BaseModel
import torch
from model import model
import pickle

app = FastAPI()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Load model
model_instance = model().to(device)
model_instance.load_state_dict(torch.load("riceClassifier.pth", map_location=device))
model_instance.eval()
with open('normalization.pkl', 'rb') as f:
    normalization_values = pickle.load(f)

# Pydantic schema for incoming request
class InputData(BaseModel):
    features: list[float]  # expects a list of 10 floats

# Inference function
def inference(input_list):
    input_tensor = torch.tensor(input_list, dtype=torch.float32)
    input_tensor = input_tensor / torch.tensor(normalization_values, dtype=torch.float32)
    input_tensor = input_tensor.unsqueeze(0).to(device)  # add batch dim and move to device
    
    with torch.no_grad():
        output = model_instance(input_tensor)
        output = round(output.item(), 2)
    return output

# API endpoint
@app.post("/predict/")
def predict(data: InputData):
    prediction = inference(data.features)
    return {"prediction": prediction}
