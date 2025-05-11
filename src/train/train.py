import os
import json
import mlflow
import torch
import torch.distributed as dist
import argparse
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from pathlib import Path
import torch.backends.cudnn as cudnn
from transformers.utils import logging
from utils import init_distributed_mode, get_rank, get_world_size, is_main_process
from utils import create_sampler, create_loader, create_dataset
from utils import save_result, coco_caption_eval
from utils import cosine_lr_schedule
from utils import train, evaluate
from utils import blip_decoder
from mlflow.tracking import MlflowClient

logger = logging.get_logger(__name__)


def train_model(args, config):

    mlflow.start_run()       
    mlflow.set_experiment('blip-captioning')
    MODEL_NAME = config.get('mlflow_model_name', 'blip-captioning')
    run_id = mlflow.active_run().info.run_id
    mlflow.log_params(config)
    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating captioning dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('caption_flickr', config)  

    if args.distributed:
        num_tasks = get_world_size()
        global_rank = get_rank()            
        samplers = create_sampler([train_dataset,val_dataset,test_dataset], [True,False,False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    print("Creating model")
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                       vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                       prompt=config['prompt'])
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
            
    best_score = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, config['max_epoch']):
        if not args.evaluate:        
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                
            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                
            train_stats = train(model, train_loader, optimizer, epoch, device) 
        
        val_result = evaluate(model_without_ddp, val_loader, device, config)  
        val_result_file = save_result(val_result, args.result_dir, 'val_epoch%d'%epoch, remove_duplicate='image_id')        
  

        if is_main_process():   
            coco_val = coco_caption_eval(config['coco_gt_root'],val_result_file,'val')
            
            current_score = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
            
            # Log metrics to MLflow
            if is_main_process():
                mlflow.log_metrics({
                    'train_loss': train_stats['loss'] if not args.evaluate else 0,
                    'val_cider': coco_val.eval['CIDEr'],
                    'val_bleu4': coco_val.eval['Bleu_4'],
                    'epoch': epoch
                }, step=epoch)
            
            if args.evaluate:            
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()},                
                            }
                with open(os.path.join(args.output_dir, "evaluate.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")                   
            else:             
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                if current_score > best_score:
                    best_score = current_score
                    best_epoch = epoch                
                    best_model_path = os.path.join(args.output_dir, 'model_base_caption_capfilt_large.pth')
                    torch.save(save_obj, best_model_path)
                    
                    # Register best model in MLflow
                    if is_main_process():
                        print(f"Best model saved and registered at epoch {epoch}")
                    
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': v for k, v in coco_val.eval.items()},                 
                             'epoch': epoch,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")     
                    
        if args.evaluate: 
            break
        dist.barrier()
    mlflow.log_artifact(
        local_path=os.path.join(args.output_dir, 'model_base_caption_capfilt_large.pth'),
        artifact_path="checkpoints"
    #     registered_model_name= "blip-model",
    #     pip_requirements=["torch", "transformers", "lightning"]
    )

    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name="blip-model")
    client.set_registered_model_alias(
        name = "blip-model",
        alias = 'development',
        version = registered_model.version,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    if is_main_process():
        mlflow.log_metric('training_time', total_time)
        mlflow.end_run()
        print("MLflow run completed")
    
    return run_id, registered_model.version

# if __name__ == '__main__':
#     namespace = dict()
#     namespace['output_dir'] = 'output/caption_flickr'
#     namespace['device'] = 'cuda'
#     namespace['seed'] = 42
#     namespace['world_size'] = 1
#     namespace['dist_url'] = 'env://'
#     namespace['distributed'] = True
#     namespace['evaluate'] = False
#     namespace['config'] = 'config.yaml'
#     args = argparse.Namespace(**namespace)

#     config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

#     args.result_dir = os.path.join(args.output_dir, 'result')

#     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
#     Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
#     yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

#     train_model(args, config)