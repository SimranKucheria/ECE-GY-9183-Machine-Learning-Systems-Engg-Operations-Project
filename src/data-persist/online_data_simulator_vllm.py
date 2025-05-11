import os
import time
import random
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

# --- Configuration ---
VLLM_SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://${FLOATING_IP}:8205/v1/completions")
LOAD_PATTERN = [int(x) for x in os.environ.get("LOAD_PATTERN", "1,2,3,5,3,2,1").split(",")]
DELAY_BETWEEN_STEPS = int(os.environ.get("DELAY_BETWEEN_STEPS", "60"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))

DATA_DIR = "/data/Flickr30k"
ONLINE_JSON = os.path.join(DATA_DIR, "flickr30k_test_online.json")

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("vllm_simulator")

# --- Data loader ---
def load_descriptions(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    samples = []
    for item in data:
        # Each item has 'caption' as a list; pick one randomly
        description = random.choice(item["caption"])
        samples.append(description)
    return samples

# --- vLLM request function ---
def request_vllm(description):
    try:
        prompt = f"""
            Generate indexing tags using the image description.
            The tags should be useful for information retrieval on an image-sharing platform.

            Example:
            Description: a man walking a dog
            Tags: man, walking, dog, pet, outdoors

            Now generate tags for the following image description: {description}
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt": prompt,
            "max_tokens": 300,
            "temperature": 0.7
        }
        logger.info(f"Hitting vLLM at {VLLM_SERVER_URL}")
        response = requests.post(VLLM_SERVER_URL, headers=headers, json=data, timeout=REQUEST_TIMEOUT)
        logger.info(f"vLLM Response status code: {response.status_code}")
        logger.info(f"vLLM response: {response.json()}")
        return response.json()
    except Exception as e:
        logger.error(f"Error during vLLM inference: {e}")
        return None

# --- Load generator ---
def send_continuous_requests(descriptions, duration_sec):
    start = time.time()
    while time.time() - start < duration_sec:
        description = random.choice(descriptions)
        request_vllm(description)
        time.sleep(0.1)

def run_load_stage(concurrent_workers, duration_sec, descriptions):
    with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
        futures = []
        for _ in range(concurrent_workers):
            futures.append(executor.submit(send_continuous_requests, descriptions, duration_sec))
        for f in futures:
            f.result()  # Wait for all to finish

# --- Main runner ---
if __name__ == "__main__":
    logger.info("Waiting for vLLM server to be ready...")
    time.sleep(10)
    
    descriptions = load_descriptions(ONLINE_JSON)
    if not descriptions:
        raise ValueError("No descriptions found in online test set.")
    
    for load in LOAD_PATTERN:
        logger.info(f"Simulating load with {load} concurrent requests...")
        run_load_stage(load, DELAY_BETWEEN_STEPS, descriptions)
        logger.info(f"Stage with {load} workers complete.")
