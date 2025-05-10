from fastapi import FastAPI
from pydantic import BaseModel, Field
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import io
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram, Counter
from alibi_detect.saving import load_detector
from transformers import ViTForImageClassification
import logging

Log = logging.getLogger("uvicorn.error")

# import onnx
# import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=2) 

# Load the change detector from file to monitor data drift
cd = load_detector("cd")

# Counter for drift events
drift_event_counter = Counter(
        'drift_events_total', 
        'Total number of drift events detected'
)

# Histogram for drift test statistic
drift_stat_hist = Histogram(
        'drift_test_stat', 
        'Drift score distribution'
)

# Histogram for prediction confidence
confidence_histogram = Histogram(
    "prediction_confidence",
    "Model prediction confidence",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
)

# Count how often we predict each class
class_counter = Counter(
    "predicted_class_total",
    "Count of predictions per class",
    ['class_name']
)

def detect_drift_async(cd, x_np):
    cd_pred = cd.predict(x_np)
    test_stat = cd_pred['data']['test_stat']
    is_drift = cd_pred['data']['is_drift']

    drift_stat_hist.observe(test_stat)
    if is_drift:
        drift_event_counter.inc()

app = FastAPI(
    title="Human VS AI Image fastAPI",
    description="API for classifying Images as Human or AI",
    version="1.0.0"
)
# Define the request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)  # Ensures probability is between 0 and 1

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the RegNet model
# MODEL_PATH = "model.pth"
# model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# model.to(device)
# model.eval()

# Load the ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
MODEL_PATH = "checkpoint_vit.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device) # load the model into the GPU
Log.info(f"Model classifier head: {model.classifier}")
Log.info(f"Number of labels: {model.config.num_labels}")
model.eval()
# model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
# model.to(device)
# model.eval()

# # ONNX model
# onnx_model_path = "../model_regnet.onnx"
# ort_session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
# print(ort_session.get_providers())

# Define class labels
classes = np.array(["Human","AI"])

# Define the image preprocessing function
# def preprocess_image(img):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return transform(img).unsqueeze(0)

# def preprocess_image(img):
#     transform = transforms.Compose([
#         transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
#     ])
#     return transform(img).unsqueeze(0)

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

@app.post("/predict")
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Preprocess the image
        image = preprocess_image(image).to(device)

        # Run inference
        with torch.no_grad():
            output = model(image).logits[:, 0]  # Get the logit
            probability = torch.sigmoid(output).item()  # Convert to probability
            predicted_class = 1 if probability >= 0.5 else 0
            confidence = torch.sigmoid(output).item()
            Log.info(f"PROBABILITY: {probability}")
            Log.info(f"PREDICTED CLASS: {predicted_class}")
            
        # # Update metrics
        # confidence_histogram.observe(confidence)
        # class_counter.labels(class_name=classes[predicted_class]).inc()

        # # Detect drift asynchronously
        # x_np = image.squeeze(0).cpu().numpy()
        # executor.submit(detect_drift_async, cd, x_np)
        return PredictionResponse(prediction=classes[predicted_class], probability=confidence)

    except Exception as e:
        return {"error": str(e)}
   
Instrumentator().instrument(app).expose(app)
