import requests
import boto3 
import os
import sys
import random
import time

LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_USER']
MINIO_SECRET_KEY = os.environ['MINIO_PASSWORD']
BUCKET_NAME = "production"
SAMPLE_SIZE = 2

# time.sleep(120)

# Function to check if Label Studio is ready
def is_label_studio_ready():
    while True:
        try:
            response = requests.get(f"{LABEL_STUDIO_URL}")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(5)
            continue
    
# Wait for Label Studio to be fully ready
print("Waiting for Label Studio to be ready...")
is_label_studio_ready()

LABEL_CONFIG = """
<View>
  <Image name="image" value="$image" maxWidth="500px"/>
  <Choices name="label" toName="image" choice="single" showInLine="true" >
    <Choice value="Human"/>
    <Choice value="AI"/>
  </Choices>
</View>
"""

headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}

# configure a project - set up its name and the appearance of the labeling interface
project_config = {
    "title": "DeepTrust Random Sample",
    "label_config": LABEL_CONFIG
}

# check if labelstudio is up
time.sleep(10)

# send it to Label Studio API
res = requests.post(f"{LABEL_STUDIO_URL}/api/projects", json=project_config, headers=headers)
if res.status_code == 201:
    PROJECT_ID = res.json()['id']
    print(f"Created new project: DeepTrust Random Sample (ID {PROJECT_ID})")
else:
    raise Exception("Failed to create project:", res.text)

s3 = boto3.client(
    "s3",
    # @TODO: Change to minio endpoint
    endpoint_url=MINIO_URL,
    # endpoint_url="http://localhost:9000/",
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name="us-east-1",
)

# get a list of objects in the “production” bucket, and randomly sample some:
all_keys = []
paginator = s3.get_paginator("list_objects_v2")
for page in paginator.paginate(Bucket=BUCKET_NAME):
    for obj in page.get("Contents", []):
        all_keys.append(obj["Key"])

sampled_keys = random.sample(all_keys, min(SAMPLE_SIZE, len(all_keys)))
# generate a URL for each object we want to label, so that the annotator can view the image from their browser
tasks = []
for key in sampled_keys:
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=3600
    )
    # and add to the list of tasks
    tasks.append({"data": {"image": presigned_url}, "meta": {"original_key": key}})

# then, send the lists of tasks to the Label Studio project
res = requests.post(
    f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import",
    json=tasks,
    headers=headers
)
if res.status_code == 201:
    print(f"Imported {len(tasks)} tasks into project {PROJECT_ID}")
    print(res.json())
else:
    raise Exception("Failed to import tasks:", res.text)