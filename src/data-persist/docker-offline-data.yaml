FROM apache/airflow:2.8.1-python3.11

USER root
RUN apt-get update && apt-get install -y gcc
USER airflow

# Copy DAGs
COPY dags/ /opt/airflow/dags/

# Copy your offline-data.py script
COPY offline-data.py /opt/airflow/offline-data.py

# Install Python dependencies
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
