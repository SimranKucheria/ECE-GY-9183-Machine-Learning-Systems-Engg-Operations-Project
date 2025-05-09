from flask import Flask, request, jsonify
import os
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)

@app.route('/update_VIT', methods=['POST'])
def upload_file():
    data = request.get_json()
    model_version = data.get('model_version')
    model_uri = f"models:/VITDeeptrustModel/{model_version}"
    client = MlflowClient()
    destination_dir = "/data/vit/model.pth"
    local_path = client.download_artifacts(None, model_uri, destination_dir)
    print(f"Model downloaded to: {local_path}")
    try:
        subprocess.run(
            ["docker-compose", "restart", "app"],
            check=True,
            cwd=os.getenv("DOCKER_COMPOSE_DIR", ".")
        )
        return jsonify({"message": "File uploaded successfully, app restarted"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Failed to restart app", "details": str(e)}), 500
    return jsonify({"message": "File uploaded successfully"}), 200

if __name__ == '__main__':
    os.makedirs("/data/vit/", exist_ok=True)
    app.run(host="0.0.0.0", port=5000)
