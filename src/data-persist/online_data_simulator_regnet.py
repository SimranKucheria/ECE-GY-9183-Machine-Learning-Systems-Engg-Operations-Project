import os
import time
import base64
import logging
import requests
import pandas as pd
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
FASTAPI_URL = os.environ.get("FASTAPI_URL", "http://${FLOATING_IP}:8000")
DATA_DIR = "/data/AiVsHuman"
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
CSV_PATH = os.path.join(DATA_DIR, "testing_online.csv")
LOAD_PATTERN = [int(x) for x in os.environ.get("LOAD_PATTERN", "1,2,3,5,3,2,1").split(",")]
DELAY_BETWEEN_STEPS = int(os.environ.get("DELAY_BETWEEN_STEPS", "60"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "5"))

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("online_data_simulator")

# --- Image utilities ---
def load_and_encode_image(image_path):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return encoded
    except Exception as e:
        logger.warning(f"Could not encode image {image_path}: {e}")
        return None

# --- Request sending ---
def send_request(image_path):
    encoded_str = load_and_encode_image(image_path)
    if not encoded_str:
        return False, None, None
    payload = {"image": encoded_str}
    try:
        url = f"{FASTAPI_URL}/predict"
        logger.info(f"Hitting FastAPI at {url}")
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        logger.info(f"FastAPI Response status code: {resp.status_code}")
        result = resp.json()
        predicted_class = result.get("prediction")
        probability = result.get("probability")
        logger.info(f"FastAPI Predicted class: {predicted_class}, Probability: {probability}")
        return True, predicted_class, probability
    except Exception as e:
        logger.error(f"Error during FastAPI inference for {image_path}: {e}")
        return False, None, None

# --- Continuous request worker ---
def send_continuous_requests(df, duration_sec):
    start = time.time()
    successes, failures = 0, 0
    while time.time() - start < duration_sec:
        row = df.sample(1).iloc[0]
        image_file = os.path.join(IMAGES_DIR, row["file_name"])
        label = row["label"]  # Not sent, but available for local evaluation if needed
        ok, predicted_class, probability = send_request(image_file)
        if ok:
            successes += 1
            # Optionally, compare predicted_class with label here for accuracy stats
        else:
            failures += 1
        time.sleep(0.1)  # Small delay to avoid hammering the server
    return successes, failures

# --- Load stage runner ---
def run_load_stage(df, concurrent_workers, duration_sec):
    logger.info(f"Starting stage: {concurrent_workers} workers for {duration_sec} seconds")
    with ThreadPoolExecutor(max_workers=concurrent_workers) as pool:
        futures = [pool.submit(send_continuous_requests, df, duration_sec) for _ in range(concurrent_workers)]
        total_success, total_failure = 0, 0
        for f in futures:
            s, f_ = f.result()
            total_success += s
            total_failure += f_
    logger.info(f"Stage done: {total_success} successes, {total_failure} failures")
    return total_success, total_failure

# --- Main runner ---
def main():
    logger.info("Loading online test data...")
    if not os.path.exists(CSV_PATH):
        logger.error(f"Online test CSV not found at {CSV_PATH}")
        return
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Loaded {len(df)} rows from {CSV_PATH}")

    total_success, total_failure = 0, 0
    for load in LOAD_PATTERN:
        s, f = run_load_stage(df, load, DELAY_BETWEEN_STEPS)
        total_success += s
        total_failure += f
        logger.info(f"Total so far: {total_success} successes, {total_failure} failures")
    logger.info("Simulation complete.")

if __name__ == "__main__":
    logger.info("Waiting 10s for FastAPI server to be ready...")
    time.sleep(10)
    main()
