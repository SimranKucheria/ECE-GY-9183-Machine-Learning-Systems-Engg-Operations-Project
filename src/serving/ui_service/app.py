import numpy as np
import requests
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import os
import base64
from mimetypes import guess_type # used to identify the type of image
from datetime import datetime # used to generate timestamp tag for image
import uuid # used to generate unique ID per image
import boto3 # client for s3-compatible object store, including MinIO
from concurrent.futures import ThreadPoolExecutor  # used for the thread pool that will upload images to MinIO
executor = ThreadPoolExecutor(max_workers=2)  # can adjust max_workers as needed
import logging
import tritonclient.http as httpclient # for making requests to Triton
from flask import jsonify 


# Authenticate to MinIO object store
s3 = boto3.client(
    's3',
    endpoint_url=os.environ['MINIO_URL'],  # e.g. 'http://minio:9000'
    aws_access_key_id=os.environ['MINIO_USER'],
    aws_secret_access_key=os.environ['MINIO_PASSWORD'],
    region_name='us-east-1'  # required for the boto client but not used by MinIO
)

app = Flask(__name__)

os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)

FASTAPI_SERVER_URL = os.environ['FASTAPI_SERVER_URL']  # New: FastAPI server URL
# FASTAPI_SERVER_URL = os.environ.get('FASTAPI_SERVER_URL', 'http://localhost:8000')

TRITON_SERVER_URL=os.environ['TRITON_SERVER_URL']

# for uploading production images to MinIO bucket
def upload_production_bucket(img_path, preds, confidence, prediction_id):
    classes = np.array(["Human","AI"])
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    pred_index = np.where(classes == preds)[0][0]
    class_dir = f"class_{pred_index:02d}"

    bucket_name = "production"
    root, ext = os.path.splitext(img_path)
    content_type = guess_type(img_path)[0] or 'application/octet-stream'
    s3_key = f"{class_dir}/{prediction_id}{ext}"
    
    with open(img_path, 'rb') as f:
        s3.upload_fileobj(f, 
            bucket_name, 
            s3_key, 
            ExtraArgs={'ContentType': content_type}
            )

    # tag the object with predicted class and confidence
    s3.put_object_tagging(
        Bucket=bucket_name,
        Key=s3_key,
        Tagging={
            'TagSet': [
                {'Key': 'predicted_class', 'Value': preds},
                {'Key': 'confidence', 'Value': f"{confidence:.3f}"},
                {'Key': 'timestamp', 'Value': timestamp}
            ]
        }
    )

# for making requests to FastAPI
def request_fastapi(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        encoded_str = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"image": encoded_str}
        
        response = requests.post(f"{FASTAPI_SERVER_URL}/predict", json=payload)
        response.raise_for_status()
        
        result = response.json()
        predicted_class = result.get("prediction")
        probability = result.get("probability")
        
        return predicted_class, probability

    except Exception as e:
        print(f"Error during inference: {e}")  
        app.logger.error(f"Error during FastAPI inference: {e}")
        return None, None 

# # Usage example
# caption = get_caption("Cat_August_2010-4.jpg")
# print(f"Generated caption: {caption}")

def request_triton(image_path):
    try:
        client = httpclient.InferenceServerClient(url="triton_server:8000")
        app.logger.info(f"IS TRITON SERVER LIVE: {client.is_server_live()}")
        with open(image_path, "rb") as f:
            image_bytes = f.read() 
        inputs = []
        inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))
        encoded_str =  base64.b64encode(image_bytes).decode("utf-8")
        input_data = np.array([[encoded_str]], dtype=object)
        inputs[0].set_data_from_numpy(input_data)
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("CAPTION", binary_data=False))
        results = client.infer(model_name="caption", inputs=inputs, outputs=outputs)
        cap = results.as_numpy("CAPTION")
        return cap
    except Exception as e:
        print(f"Error during Triton inference: {e}")  
        return None, None  
    
def request_vllm(description):
    try:
        url = "http://192.5.87.164:8005/v1/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "mistralai/Mistral-7B-Instruct-v0.2",
            "prompt": "Generate 5 tags for the description: " + description,
            "max_tokens": 32,
            "temperature": 0.7
        }
        response = requests.post(url, headers=headers, json=data)
        print(response.json())
    except Exception as e:
        print(f"Error during vLLM inference: {e}")  
        app.logger.error(f"Error during inference: {e}")
        return None, None 

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    preds = None
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.instance_path, 'uploads', secure_filename(f.filename)))
        img_path = os.path.join(app.instance_path, 'uploads', secure_filename(f.filename))

        # create a unique filename for the image
        prediction_id = str(uuid.uuid4())
        app.logger.info(f"Prediction ID: {prediction_id}")

        preds, probs = request_fastapi(img_path)
        description = request_triton(img_path)
        tags = request_vllm(description)

        app.logger.info(f"Predicted class: {preds}")
        app.logger.info(f"Description: {description}")
        app.logger.info(f"Tags: {tags}")

        if preds:
            executor.submit(upload_production_bucket, img_path, preds, probs, prediction_id) # New! upload production image to MinIO bucket
            app.logger.info(f"Image uploaded to production bucket with ID: {prediction_id}")
            # return f'<button type="button" class="btn btn-info btn-sm">{preds}</button>'
            return f'''
                <div class="alert alert-info">
                    <h5>Prediction:</h5>
                    <p>{preds}</p>
                    <h5>Description:</h5>
                    <p>{description}</p>
                    <h5>Tags:</h5>
                    <p>{tags}</p>
                </div>
            '''
    return '<div class="alert alert-warning">Warning: No prediction made.</div>'

@app.route('/test', methods=['GET'])
def test():
    img_path = os.path.join(app.instance_path, 'uploads', 'test_image.jpeg')
    preds, probs = request_fastapi(img_path)
    return str(preds)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
