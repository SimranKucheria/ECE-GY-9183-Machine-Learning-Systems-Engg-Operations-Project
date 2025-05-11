import os
import time
import random
import base64
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from io import BytesIO

import tritonclient.http as httpclient

TRITON_SERVER_URL = os.environ.get("TRITON_SERVER_URL", "http://${FLOATING_IP}:8000/triton")
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

def request_triton(image_path, caption=None):
    """Send request to Triton Inference Server for image captioning"""
    try:
        client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL, verbose=False)
        if not client.is_server_live():
            print("Triton server is not live.")
            return None
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        inputs = []
        inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        input_data = np.array([[encoded_str]], dtype=object)
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("CAPTION", binary_data=False))
        
        results = client.infer(
            model_name="caption",
            inputs=inputs,
            outputs=outputs,
            timeout=REQUEST_TIMEOUT
        )
        cap = results.as_numpy("CAPTION")
        generated_caption = cap[0] if cap is not None and len(cap) > 0 else None
        
        print(f"Triton generated caption: {generated_caption}")
        print(f"Original caption: {caption}")
        return generated_caption
    except Exception as e:
        import traceback
        print(f"Triton inference failed: {e}")
        print(traceback.format_exc())
        return None

def send_continuous_requests(samples, duration_sec):
    start = time.time()
    while time.time() - start < duration_sec:
        image_path, caption = random.choice(samples)
        request_triton(image_path, caption)
        time.sleep(0.1)

def run_load_stage(concurrent_workers, duration_sec, samples):
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        for _ in range(concurrent_workers):
            executor.submit(send_continuous_requests, samples, duration_sec)

if __name__ == "__main__":
    print("Waiting for Triton server to be ready...")
    time.sleep(10)
    
    samples = load_image_paths_and_captions(ONLINE_JSON)
    if not samples:
        raise ValueError("No valid images found in online test set.")
    
    for load in LOAD_PATTERN:
        print(f"Simulating load with {load} concurrent requests...")
        run_load_stage(load, DELAY_BETWEEN_STEPS, samples)
        print(f"Stage with {load} workers complete.")
