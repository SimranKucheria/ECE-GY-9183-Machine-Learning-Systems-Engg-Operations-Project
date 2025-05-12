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
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

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
    nproc_per_node: Optional[int] = 1




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
        nproc_per_node = request.nproc_per_node

        if namespace['evaluate']:
            cmd = [
                "python", "-m", "torch.distributed.run",
                "--nproc_per_node={}".format(nproc_per_node),
                "train.py",
                "--evaluate"
            ]
        else:
            cmd = [
                "python", "-m", "torch.distributed.run",
                "--nproc_per_node={}".format(nproc_per_node),
                "train.py"
            ]

        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)  # Match your desired indentation
        
        config = CommentedMap()  # Ordered dict with comment support

        # run_id, version = train_model(args, config)
        # logger.info(f"Training job started with run_id: {run_id} and version: {version}")

        # response = {
        #     "status": "success",
        #     "run_id": run_id,
        #     "version": version,
        #     "output_dir": request.output_dir
        # }
        # logger.info("Train job completed successfully")
        # return response

        #python -m torch.distributed.run --nproc_per_node=8 train_caption.py 
        config['image_root'] = request.image_root
        config['ann_root'] = request.ann_root
        config['coco_gt_root'] = request.coco_gt_root
        
        # Add comment before 'pretrained'
        config.yaml_set_comment_before_after_key(
            'pretrained', 
            before='\n# set pretrained as a file path or an url'
        )
        config['pretrained'] = request.pretrained
        
        # Add comment before 'vit'
        config.yaml_set_comment_before_after_key(
            'vit', 
            before='\n# size of vit model; base or large'
        )
        config['vit'] = request.vit
        config['vit_grad_ckpt'] = request.vit_grad_ckpt
        config['vit_ckpt_layer'] = request.vit_ckpt_layer
        config['batch_size'] = request.batch_size
        config['init_lr'] = request.init_lr
        
        # Add commented alternatives before 'image_size'
        config.yaml_set_comment_before_after_key(
            'image_size',
            before='\n# vit: \'large\'\n# vit_grad_ckpt: True\n# vit_ckpt_layer: 5\n# batch_size: 16\n# init_lr: 2e-6'
        )
        config['image_size'] = request.iamge_size
        
        # Generation configs
        config.yaml_set_comment_before_after_key(
            'max_length', 
            before='\n# generation configs'
        )
        config['max_length'] = request.max_length
        config['min_length'] = request.min_length
        config['num_beams'] = request.num_beams
        config['prompt'] = request.prompt
        
        # Optimizer
        config.yaml_set_comment_before_after_key(
            'weight_decay', 
            before='\n# optimizer'
        )
        config['weight_decay'] = request.weight_decay
        config['min_lr'] = request.min_lr
        config['max_epoch'] = request.max_epoch

        

        output_path = "config.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(config, f)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        time.sleep(5)  
        with open("/mnt/data/results_blip.json", "r") as f:
            saved_file_respoonse = json.load(f)

        response = saved_file_respoonse
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