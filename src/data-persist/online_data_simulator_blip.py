import os
import time
import random
import base64
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from io import BytesIO

FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://${FLOATING_IP}:8000/")
LOAD_PATTERN = [int(x) for x in os.environ.get("LOAD_PATTERN", "1,2,3,5,3,2,1").split(",")]
DELAY_BETWEEN_STEPS = int(os.environ.get("DELAY_BETWEEN_STEPS", "60"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))

DATA_DIR = "/data/Flickr30k"
IMAGES_DIR = os.path.join(DATA_DIR, "flickr30k-images")
ONLINE_JSON = os.path.join(DATA_DIR, "flickr30k_test_online.json")

def load_image_paths_and_captions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    samples = []
    for item in data:
        image_path = os.path.join(IMAGES_DIR, Path(item["image"]).name)
        if os.path.exists(image_path):
            samples.append((image_path, random.choice(item["caption"])))
    return samples

def encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded
    except Exception as e:
        print(f"Could not encode image {image_path}: {e}")
        return None

def send_request(encoded_img, caption):
    payload = {"image": encoded_img, "caption": caption}
    try:
        resp = requests.post(FASTAPI_URL, json=payload, timeout=REQUEST_TIMEOUT)
        print(f"Status: {resp.status_code}, Result: {resp.text}")
    except Exception as e:
        print(f"Failed request: {e}")

def send_continuous_requests(samples, duration_sec):
    start = time.time()
    while time.time() - start < duration_sec:
        image_path, caption = random.choice(samples)
        encoded_img = encode_image(image_path)
        if encoded_img:
            send_request(encoded_img, caption)
        time.sleep(0.1)

def run_load_stage(concurrent_workers, duration_sec, samples):
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        for _ in range(concurrent_workers):
            executor.submit(send_continuous_requests, samples, duration_sec)

if __name__ == "__main__":
    print("Waiting for FastAPI server to be ready...")
    time.sleep(10)
    samples = load_image_paths_and_captions(ONLINE_JSON)
    if not samples:
        raise ValueError("No valid images found in online test set.")
    for load in LOAD_PATTERN:
        print(f"Simulating load with {load} concurrent requests...")
        run_load_stage(load, DELAY_BETWEEN_STEPS, samples)
        print(f"Stage with {load} workers complete.")
