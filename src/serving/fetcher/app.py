from flask import Flask, request, jsonify
import os
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)

@app.route('/update_VIT', methods=['POST'])
def upload_file():
    app.logger.info("Entered app")
    data = request.get_json()
    model_version = data.get('model_version')
    client = MlflowClient()
    destination_dir = "/data/vit/"
    app.logger.info("Dowloading Model")
    try:
        os.remove("/data/vit/model/data/model.pth")
        print("/data/vit/model/data/model.pth deleted successfully.")
    except FileNotFoundError:
        pass
    local_path = client.download_artifacts(model_version, "model/data/model.pth", destination_dir)
    app.logger.info(f"Model downloaded to: {local_path}")
    file_path = "../scripts/"+os.environ["DOCKER_COMPOSE_DIR"]
    app.logger.info(file_path)
    
    my_env = os.environ.copy()
    

    print(f"Error: File '{file_path}' not found.")
    try:
        command = [ 
        "docker", "compose",
        "-f", file_path,
        "up", "--build", "-d", "--force-recreate"]
        subprocess.run(command,env=my_env)
    
        return jsonify({"message": "File uploaded successfully, app restarted"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Failed to restart app", "details": str(e)}), 500
    return jsonify({"message": "File uploaded successfully"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000,debug=True)
