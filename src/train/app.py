from fastapi import FastAPI, HTTPException
import subprocess
import logging
from typing import Dict

app = FastAPI()
logger = logging.getLogger("ray-train-server")
logger.setLevel(logging.INFO)

@app.post("/start-training-vit")
def start_training_job() -> Dict:
    """Endpoint to submit Ray training job"""
    try:
        # Build the ray job submit command
        cmd = [
            "ray", "job", "submit",
            "--runtime-env", "runtime_ray.json",
            "--working-dir", ".",
            "--", "python", "aivshuman/train.py"
        ]
        
        logger.info(f"Starting Ray job with command: {' '.join(cmd)}")
        
        # Execute command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        response = {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        logger.info("Ray job completed successfully")
        return response

    except subprocess.CalledProcessError as e:
        logger.error(f"Ray job failed: {e.stderr}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Training job failed",
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