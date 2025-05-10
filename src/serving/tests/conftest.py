import os
import json
import base64
import random
import requests
import numpy as np
import pandas as pd
import pytest
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
from torchsummary import summary
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import requests
import tritonclient.http as httpclient
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


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
    model_path = "/Users/manali/nyu/COURSES/Sem4/MLOps/serving/inference_service/model/data/model.pth"  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    _ = model.eval()  
    return model


# function that returns the val loader
@pytest.fixture(scope="session")
def test_data():
    # @TODO: replace with the actual path
    csv_file = pd.read_csv("/mnt/data/AiVsHuman/validation.csv")
    img_dir="/mnt/data/AiVsHuman/Images"
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

@pytest.fixture(scope="session")
def triton_client():
    return httpclient.InferenceServerClient(url="triton_server:8000")

@pytest.fixture(scope="session")
def test_data_BLIP():
    json_file = pd.read_csv('/mnt/data/Flickr30k/flickr30k_val.json')
    img_dir="/mnt/data/Flickr30k"

    with open(json_file) as f:
        dataset = json.load(f)

    data = []
    for entry in dataset:
        img_name = entry['image']
        img_path = os.path.join(img_dir, img_name)
        gt_captions = entry['caption']  # list of 5 captions
    
        try:
            # Check if image exists
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
        
            # If image exists, append the data
            data.append({'image_path': img_path, 'expected_caption': gt_captions})
            # print("appended")
    
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue  # Skip this image and move to the next one                     
    return data

@pytest.fixture(scope="session")
def get_caption(triton_client):
    def _get_caption(image_path):
        with open(image_path, "rb") as f:
            image_bytes = f.read() 
        inputs = []
        inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        input_data = np.array([[encoded_str]], dtype=object)
        inputs[0].set_data_from_numpy(input_data)
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("CAPTION", binary_data=False))
        results = triton_client.infer(model_name="caption", inputs=inputs, outputs=outputs)
        cap = results.as_numpy("CAPTION")
        return cap
    return _get_caption

@pytest.fixture(scope="session")
def generate_all_captions(test_data_BLIP, get_caption):
    results = []
    for sample in test_data_BLIP:
        image_path = sample['image_path']
        expected = sample['expected_caption']
        generated = get_caption(image_path)
        results.append({
            'image_path': image_path,
            'expected_caption': expected,
            'generated_caption': generated
        })
    return results

@pytest.fixture(scope="session")
def sample_image_path():
    # Return sample image path for testing captions
    return "/mnt/data/Flickr30k/flickr30k_images/1009434119.jpg" 