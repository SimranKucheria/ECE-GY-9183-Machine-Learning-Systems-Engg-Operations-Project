import base64
import pytest
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import requests


# Custom dataset class to load and preprocess images from human VS AI Images
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.annotations.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        return image, label

# function to load the train and validation folders
# @pytest.fixture(scope="session")
def create_dataloaders(csv_file, img_dir, img_size=(224, 224), batch_size=32, n_fold=0):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])

    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(len(csv_file)), csv_file.iloc[:, 1].values)):
        if i == n_fold:
            break

    train_dataset = Subset(dataset, train_index)
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.ToTensor()
                               ]))
    val_dataset = Subset(dataset, val_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

# @pytest.fixture(scope="session")
# def transform():
#     transform = transforms.Compose([
#         transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
#     return transform().unsqueeze(0) 


# function that returns the predictions of the RegNet model on val loader
@pytest.fixture(scope="session")
def predict(transform):
    def predict_image(model, image):
        model.eval()
        with torch.no_grad():
            input_tensor = transform(image)
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
            predicted_class = torch.argmax(probabilities, 1).item()
            confidence = probabilities[0, predicted_class].item()  # Get the probability
            return predicted_class
    return predict_image

@pytest.fixture(scope="session")
def model():
    model_path = "/Users/manali/nyu/COURSES/Sem4/MLOps/serving/inference_service/model.pth"  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    _ = model.eval()  
    return model


# function that returns the val loader
@pytest.fixture(scope="session")
def test_data():
    # @TODO: replace with the actual path
    csv_file = pd.read_csv("/Users/manali/nyu/COURSES/Sem4/MLOps/serving/AiVsHuman/validation.csv")
    img_dir="/Users/manali/nyu/COURSES/Sem4/MLOps/serving/AiVsHuman/Images"
    train_loader, val_loader = create_dataloaders(
        csv_file=csv_file,
        img_dir=img_dir,
        img_size=(224,224),
        batch_size=32,
        n_fold=0
    )
    return val_loader

# function that returns the predictions of the RegNet model on val loader
# @TODO: replace with ViT model
@pytest.fixture(scope="session")
def predictions(model, test_data):
    dataset_size = len(test_data.dataset)
    all_predictions = np.empty(dataset_size, dtype=np.int64)
    all_labels = np.empty(dataset_size, dtype=np.int64)

    current_index = 0
    with torch.no_grad():
        for images, labels in test_data:
            tqdm.write(f"Processing batch with {len(images)} images")
            batch_size = labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions[current_index:current_index + batch_size] = predicted.cpu().numpy()
            all_labels[current_index:current_index + batch_size] = labels.cpu().numpy()
            current_index += batch_size

    return all_labels, all_predictions

# Placeholder: Adjust the Triton server endpoint
@pytest.fixture(scope="session")
def triton_server_url():
    return "http://localhost:8000/v2/models/blip_model/infer"

@pytest.fixture(scope="module")
def test_data_BLIP():
    # Dummy test data for image paths and expected captions
    # Replace with actual image paths and expected captions
    data = [
        {"image_path": "./captioning_dummy_test_images/10002456.jpg", "expected_caption": "Workers look down from up above on a piece of equipment ."},
        {"image_path": "./captioning_dummy_test_images/1000092795.jpg", "expected_caption": "Two men in green shirts are standing in a yard ."},
    ]                     

    return data

@pytest.fixture(scope="module")
def generate_caption_triton(image_path, triton_server_url):
    with open(image_path, 'rb') as img_file:
        img_bytes = img_file.read()

    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    payload = {
        "inputs": [
            {
                "name": "image",
                "shape": [1],
                "datatype": "BYTES",
                "data": [img_base64]
            }
        ]
    }
    response = requests.post(triton_server_url, json=payload)
    response.raise_for_status()
    result = response.json()
    caption = result['outputs'][0]['data'][0]
    return caption