from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import os
import requests

# --- ENVIRONMENT VARIABLES ---
LABEL_STUDIO_URL = os.environ['LABEL_STUDIO_URL']
LABEL_STUDIO_TOKEN = os.environ['LABEL_STUDIO_USER_TOKEN']
MINIO_URL = os.environ['MINIO_URL']
MINIO_ACCESS_KEY = os.environ['MINIO_USER']
MINIO_SECRET_KEY = os.environ['MINIO_PASSWORD']

PROJECT_NAME = "DeepTrust"  # Updated to match your latest script

# --- DEFAULT ARGS ---
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def ensure_bucket(bucket, s3):
    """Ensure the target bucket exists in MinIO."""
    existing = {b['Name'] for b in s3.list_buckets()['Buckets']}
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)

def get_label_studio_results(**context):
    """Fetch completed tasks from Label Studio for the specified project."""
    headers = {"Authorization": f"Token {LABEL_STUDIO_TOKEN}"}
    response = requests.get(f"{LABEL_STUDIO_URL}/api/projects", headers=headers)
    response.raise_for_status()
    projects = response.json().get("results", [])
    project_id = next((p['id'] for p in projects if p['title'] == PROJECT_NAME), None)

    if not project_id:
        raise Exception(f"Label Studio project '{PROJECT_NAME}' not found.")

    tasks_response = requests.get(
        f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks?completed=true",
        headers=headers
    )
    tasks_response.raise_for_status()
    completed_tasks = tasks_response.json()
    context['ti'].xcom_push(key='completed_tasks', value=completed_tasks)

def process_labeled_data_minio(**context):
    """Copy processed files to MinIO buckets and preserve tags."""
    s3 = boto3.client(
        's3',
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1"
    )
    ensure_bucket("production-clean", s3)
    ensure_bucket("production-noisy", s3)

    completed_tasks = context['ti'].xcom_pull(key='completed_tasks', task_ids='get_label_studio_results')
    for task in completed_tasks:
        try:
            original_key = task["meta"]["original_key"]
            filename = original_key.split("/")[-1]
            new_key = filename  # No class directory

            # Copy to production-clean
            copy_source = {'Bucket': 'production-label-wait', 'Key': original_key}
            s3.copy_object(
                Bucket='production-clean',
                CopySource=copy_source,
                Key=new_key
            )

            # Copy to production-noisy
            s3.copy_object(
                Bucket='production-noisy',
                CopySource=copy_source,
                Key=new_key
            )

            # Preserve tags
            tags = s3.get_object_tagging(
                Bucket='production-label-wait', Key=original_key
            )['TagSet']
            s3.put_object_tagging(
                Bucket='production-clean',
                Key=new_key,
                Tagging={'TagSet': tags}
            )

            # Delete from production-label-wait
            try:
                s3.head_object(Bucket='production-label-wait', Key=original_key)
                s3.delete_object(Bucket='production-label-wait', Key=original_key)
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print(f"Object not found for deletion: {original_key}")
                else:
                    raise

            print(f"Processed {original_key} → '{new_key}'")

        except Exception as e:
            print(f"Error processing task {task.get('id')}: {e}")

with DAG(
    dag_id="pipeline_2_process_labeled_data",
    default_args=default_args,
    start_date=datetime.today() - timedelta(days=1),
    schedule_interval="@daily",
    catchup=False,
    description="Process data from Label Studio and move to MinIO buckets",
) as dag:

    get_results_task = PythonOperator(
        task_id="get_label_studio_results",
        python_callable=get_label_studio_results,
        provide_context=True,
    )

    process_task = PythonOperator(
        task_id="process_labeled_data_minio",
        python_callable=process_labeled_data_minio,
        provide_context=True,
    )

    get_results_task >> process_task
