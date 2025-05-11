from fastapi import FastAPI, HTTPException
import subprocess
import logging
from typing import Dict, Optional, List
from pydantic import BaseModel
import json
import re
import time
app = FastAPI()
logger = logging.getLogger("ray-train-server")
logger.setLevel(logging.INFO)


class TrainingRequest(BaseModel):
    model_name: Optional[str] = "google/vit-base-patch16-224"
    img_size: Optional[List[int]] = [224, 224]
    batch_size: Optional[int] = 32
    lr: Optional[float] = 2e-5
    num_epochs: Optional[int] = 10
    warmup_epochs: Optional[int] = 0
    n_fold: Optional[int] = 0
    num_workers: Optional[int] = 1
    run_name: Optional[str] = "default_run"
    mlflow_model_name: Optional[str] = "VITDeeptrustModel"
    fsdp: Optional[bool] = False


# @app.post("/start-training-vit")
# async def start_training_job(request: TrainingRequest) -> Dict:
#     """Endpoint to submit Ray training job with configurable parameters"""
#     try:
#         # Build the command with arguments
#         base_cmd = [
#             "ray", "job", "submit",
#             "--runtime-env", "runtime_ray.json",
#             "--working-dir", ".",
#             "--", "python", "aivshuman/train.py"
#         ]
        
#         # Add arguments from the request
#         args = [
#             f"--model-name={request.model_name}",
#             f"--img-size={request.img_size[0]} {request.img_size[1]}",
#             f"--batch-size={request.batch_size}",
#             f"--lr={request.lr}",
#             f"--num-epochs={request.num_epochs}",
#             f"--warmup-epochs={request.warmup_epochs}",
#             f"--n-fold={request.n_fold}",
#             f"--num-workers={request.num_workers}",
#             f"--run-name={request.run_name}",
#             f"--mlflow-model-name={request.mlflow_model_name}"
#         ]
        
#         cmd = base_cmd + args
        
#         logger.info(f"Starting Ray job with command: {' '.join(cmd)}")
        
#         # Execute command and capture output
#         result = subprocess.run(
#             cmd,
#             capture_output=True,
#             text=True,
#             check=True
#         )
        
#         response = {
#             "status": "success",
#             "command": ' '.join(cmd),
#             "stdout": result.stdout,
#             "stderr": result.stderr
#         }
#         logger.info("Ray job completed successfully")
#         return response

#     except subprocess.CalledProcessError as e:
#         logger.error(f"Ray job failed: {e.stderr}")
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "status": "error",
#                 "message": "Training job failed",
#                 "command": ' '.join(cmd),
#                 "stdout": e.stdout,
#                 "stderr": e.stderr
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail={"status": "error", "message": str(e)}
#         )


@app.post("/start-training-vit")
async def start_training_job(request: TrainingRequest) -> Dict:
    try:
        # Build the command with arguments
        base_cmd = [
            "ray", "job", "submit",
            "--runtime-env", "runtime_ray.json",
            "--working-dir", ".",
            "--", "python", "aivshuman/train.py"
        ]
        
        # Add arguments from the request
        if request.fsdp:
            args = [
                f"--model-name={request.model_name}",
                # f"--img-size={request.img_size[0]} {request.img_size[1]}",
                f"--batch-size={request.batch_size}",
                f"--lr={request.lr}",
                f"--num-epochs={request.num_epochs}",
                f"--warmup-epochs={request.warmup_epochs}",
                f"--n-fold={request.n_fold}",
                f"--num-workers={request.num_workers}",
                f"--run-name={request.run_name}",  # Add these if they're in your TrainingRequest
                f"--mlflow-model-name={request.mlflow_model_name}",
                "--fsdp",
            ]
        else:
            args = [
                f"--model-name={request.model_name}",
                # f"--img-size={request.img_size[0]} {request.img_size[1]}",
                f"--batch-size={request.batch_size}",
                f"--lr={request.lr}",
                f"--num-epochs={request.num_epochs}",
                f"--warmup-epochs={request.warmup_epochs}",
                f"--n-fold={request.n_fold}",
                f"--num-workers={request.num_workers}",
                f"--run-name={request.run_name}",  # Add these if they're in your TrainingRequest
                f"--mlflow-model-name={request.mlflow_model_name}"
            ]
        
        cmd = base_cmd + args
        
        logger.info(f"Starting Ray job with command: {' '.join(cmd)}")
        
        # Execute command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output to get the results (assuming Ray outputs JSON)
        # try:
        #     # The actual output might be in stdout or stderr depending on Ray's configuration
        #     # output = result.stdout.strip().split('\n')[-1]  # Get last line which should contain the JSON
        #     training_results = extract_json_from_stdout(result.stdout)
        # except (json.JSONDecodeError, IndexError) as e:
        #     logger.warning(f"Could not parse training results: {str(e)}")
        #     training_results = {}
        
        # response = {
        #     "status": "success",
        #     "command": ' '.join(cmd),
        #     "stdout": result.stdout,
        #     "stderr": result.stderr,
        #     "training_results": training_results
        # }
        # Read data from the file

        time.sleep(5)  
        with open("/mnt/data/results_vit.json", "r") as f:
            saved_file_respoonse = json.load(f)

        response = saved_file_respoonse
        logger.info("Ray job completed successfully")
        return response

    except subprocess.CalledProcessError as e:
        logger.error(f"Ray job failed: {e.stderr}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Training job failed",
                "command": ' '.join(cmd),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)}
        )

@app.get("/ping")
def ping() -> Dict:
    """Endpoint to check if the server is running"""
    return {"status": "ok", "message": "Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)