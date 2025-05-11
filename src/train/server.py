from fastapi import FastAPI, HTTPException
import subprocess
import logging
from typing import Dict, Optional, List
from pydantic import BaseModel
import json
import re
import time
from train import train_model
import argparse


app = FastAPI()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrainingRequest(BaseModel):
    output_dir: Optional[str] = "output/caption_flickr"
    device: Optional[str] = "cuda"
    seed: Optional[int] = 42
    world_size: Optional[int] = 1
    dist_url: Optional[str] = "env://"
    distributed: Optional[bool] = True
    evaluate: Optional[bool] = False
    image_root: Optional[str] = "/mnt/data/Flickr30k"
    ann_root: Optional[str] = "/mnt/data/Flickr30k"
    coco_gt_root: Optional[str] = "/mnt/data/Flickr30k"
    pretrained: Optional[str] = "google/vit-base-patch16-224"
    vit: Optional[str] = "base"
    vit_grad_ckpt: Optional[bool] = False
    vit_ckpt_layer: Optional[int] = 0
    batch_size: Optional[int] = 32
    init_lr: Optional[float] = 2e-5
    iamge_size: Optional[int] = 384
    max_length: Optional[int] = 20
    min_length: Optional[int] = 5
    num_beams: Optional[int] = 3
    prompt: Optional[str] = "a picture of"
    weight_decay: Optional[float] = 0.05
    min_lr: Optional[float] = 0.0
    max_epoch: Optional[int] = 5




@app.post("/start-training-blip")
async def start_training_job(request: TrainingRequest) -> Dict:
    try:
        namespace = dict()
        namespace['output_dir'] = request.output_dir
        namespace['device'] = request.device
        namespace['seed'] = request.seed
        namespace['world_size'] = request.world_size
        namespace['dist_url'] = request.dist_url
        namespace['distributed'] = request.distributed
        namespace['evaluate'] = request.evaluate
        args = argparse.Namespace(**namespace)

        config = dict()
        config['image_root'] = request.image_root
        config['ann_root'] = request.ann_root
        config['coco_gt_root'] = request.coco_gt_root
        config['pretrained'] = request.pretrained
        config['vit'] = request.vit
        config['vit_grad_ckpt'] = request.vit_grad_ckpt
        config['vit_ckpt_layer'] = request.vit_ckpt_layer
        config['batch_size'] = request.batch_size
        config['init_lr'] = request.init_lr
        config['image_size'] = request.iamge_size
        config['max_length'] = request.max_length
        config['min_length'] = request.min_length
        config['num_beams'] = request.num_beams
        config['prompt'] = request.prompt
        config['weight_decay'] = request.weight_decay
        config['min_lr'] = request.min_lr
        config['max_epoch'] = request.max_epoch

        run_id, version = train_model(args, config)
        logger.info(f"Training job started with run_id: {run_id} and version: {version}")

        response = {
            "status": "success",
            "run_id": run_id,
            "version": version,
            "output_dir": request.output_dir
        }
        logger.info("Ray job completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error starting training job: {str(e)}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/ping")
def ping() -> Dict:
    """Endpoint to check if the server is running"""
    return {"status": "ok", "message": "Server is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)