import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from PIL import Image
import os
from glob import glob
from transformers import get_cosine_schedule_with_warmup, ViTForImageClassification

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
import ray.train
import ray.train.lightning


import mlflow
from mlflow.tracking import MlflowClient
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
import numpy as np
import json

# ray.init()
# mlflow.set_tracking_uri("http://127.0.0.1:8080")

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = csv_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(self.annotations.iloc[idx, 1]))
        if self.transform:
            image = self.transform(image)
        return image, label
    
def create_dataloaders(csv_file, img_dir, img_size=(224, 224), batch_size=32, n_fold=0):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(), 
        transforms.RandomHorizontalFlip()
    ])

    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, transform=transform)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)
    for i, (train_index, val_index) in enumerate(skf.split(np.zeros(len(csv_file)), csv_file.iloc[:, 1].values)):
        if i == n_fold:
            break
            
    train_dataset = Subset(dataset, train_index)
    dataset = CustomImageDataset(csv_file=csv_file, img_dir=img_dir, 
                               transform=transforms.Compose([
                                   transforms.Resize(img_size), 
                                   transforms.ToTensor()
                               ]))
    val_dataset = Subset(dataset, val_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def train_func(config):
    # Set up MLFlow
    mlflow_logger = MLFlowLogger(
        experiment_name="vit-ai-detection",
        tracking_uri=mlflow.get_tracking_uri(),
        run_name=config["run_name"]
    )

    mlflow.start_run(run_id=mlflow_logger.run_id)

    config["labels"] = pd.read_csv(config["train_csv_path"]).iloc[:, 1:].copy()

    # Preparing data
    train_loader, val_loader = create_dataloaders(
        csv_file=config["labels"],
        img_dir=config["img_dir"],
        img_size=config["img_size"],
        batch_size=config["batch_size"],
        n_fold=config["n_fold"]
    )

    # Model
    class LitViTModel(L.LightningModule):
        def __init__(self, model_name, lr=2e-5, warmup_epochs=0):
            super().__init__()
            self.model = ViTForImageClassification.from_pretrained(model_name)
            self.criterion = nn.BCEWithLogitsLoss()
            self.lr = lr
            self.warmup_epochs = warmup_epochs
            
            self.train_acc = BinaryAccuracy()
            self.val_acc = BinaryAccuracy()
            self.val_f1 = BinaryF1Score()
            self.val_auc = BinaryAUROC()

        def forward(self, x):
            return self.model(x).logits[:, :1]

        def training_step(self, batch, batch_idx):
            x, y = batch
            y = y.float().unsqueeze(1)
            logits = self(x)
            loss = self.criterion(logits, y)
            self.log("train_loss", loss, prog_bar=True)
            self.train_acc(torch.sigmoid(logits), y)
            self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y = y.float().unsqueeze(1)
            logits = self(x)
            loss = self.criterion(logits, y)
            
            probs = torch.sigmoid(logits)
            self.val_acc(probs, y)
            self.val_f1(probs, y)
            self.val_auc(probs, y)
            
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_acc, prog_bar=True)
            self.log("val_f1", self.val_f1, prog_bar=True)
            self.log("val_auc", self.val_auc, prog_bar=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.trainer.estimated_stepping_batches * self.warmup_epochs,
                num_training_steps=self.trainer.estimated_stepping_batches * self.trainer.max_epochs
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    # Model
    model = LitViTModel(
        model_name=config["model_name"],
        lr=config["lr"],
        warmup_epochs=config["warmup_epochs"]
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_f1",
        patience=3,
        mode="max",
        verbose=True
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename="best-checkpoint"
    )
    
    # Trainer
    if config["fsdp"]:
        strategy = ray.train.lightning.RayFSDPStrategy(
            fsdp_config={"fsdp": {"sharding_strategy": "FULL_SHARD"}}
        )
    else:
        strategy = ray.train.lightning.RayDDPStrategy()
    trainer = L.Trainer(
        logger=mlflow_logger,
        callbacks=[early_stop, checkpoint_callback, ray.train.lightning.RayTrainReportCallback()],
        max_epochs=config["num_epochs"],
        accelerator="auto",
        devices="auto",
        enable_progress_bar=True,
        log_every_n_steps=10,
        strategy=strategy,
        plugins = [ray.train.lightning.RayLightningEnvironment()]
    )

    # Log parameters
    mlflow_logger.log_hyperparams(config)
    
    # Train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_model = LitViTModel.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model_name=config["model_name"],
        lr=config["lr"],
        warmup_epochs=config["warmup_epochs"]
    )

    MODEL_NAME = config["mlflow_model_name"]

    mlflow.pytorch.log_model(
        pytorch_model=best_model,
        artifact_path="model",
    #     registered_model_name=MODEL_NAME,
    #     pip_requirements=["torch", "transformers", "lightning"]
    )
    client = MlflowClient()
    run_id = mlflow.active_run().info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client.set_registered_model_alias(
        name =MODEL_NAME,
        alias = 'development',
        version = registered_model.version,
    )

    results = {
        "status": "success",
        "new_model_version": run_id,
        "model_name": MODEL_NAME,
        "mv": registered_model.version, 
        "model_uri": model_uri,
    }
    
    mlflow.end_run()


    print(results)

    # print(results)
    # Create directory if it doesn't exist
    # os.makedirs("/mnt/data", exist_ok=True)
    with open("/mnt/data/results_vit.json", "w") as f:
        json.dump(results, f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT model for AI detection")
    

    parser.add_argument("--model-name", type=str, default="google/vit-base-patch16-224",
                       help="Pretrained model name from HuggingFace")
    
    parser.add_argument("--img-size", type=int, nargs=2, default=[224, 224],
                       help="Image size (height width)")
    
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    
    parser.add_argument("--num-epochs", type=int, default=2,
                       help="Number of training epochs")
    
    parser.add_argument("--warmup-epochs", type=int, default=0,
                       help="Number of warmup epochs")
    
    parser.add_argument("--n-fold", type=int, default=0,
                       help="Fold number for cross-validation (0-4)")
    
    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel training workers")
    
    parser.add_argument("--run-name", type=str, default="default_run",
                       help="Name of the MLFlow run")
    
    parser.add_argument("--mlflow-model-name", type=str, default="VITDeeptrustModel",
                       help="Name of the MLFlow model")
    parser.add_argument("--fsdp", action="store_true",
                       help="Use Fully Sharded Data Parallel (FSDP) for distributed training")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Configuration
    # ray.init(
    #     num_cpus=6,  # Adjust based on your Mac's CPU cores
    #     include_dashboard=True,  # Disable dashboard to reduce overhead
    #     ignore_reinit_error=True
    # )
    args = parse_args()

    # print("Ray cluster resources:", ray.cluster_resources())
    files = os.listdir("/")
    print(files)

    data_dir = os.getenv("AIVSHUMAN_DATA", "/mnt/data/AiVsHuman")
    train_csv_path = os.path.join(data_dir, "train.csv")
    # config = {
    #     # "labels": pd.read_csv(train_csv_path).iloc[:, 1:].copy(),
    #     "train_csv_path": train_csv_path,
    #     "img_dir": data_dir,
    #     "model_name": "google/vit-base-patch16-224",
    #     "img_size": (224, 224),
    #     "batch_size": 32,  # Reduced for local execution
    #     "lr": 2e-5,
    #     "num_epochs": 2,
    #     "warmup_epochs": 0,
    #     "n_fold": 0,
    #     "num_workers": 1  # Number of parallel training workers
    # }

    config = {
        "train_csv_path": train_csv_path,
        "img_dir": data_dir,
        "model_name": args.model_name,
        "img_size": tuple(args.img_size),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "warmup_epochs": args.warmup_epochs,
        "n_fold": args.n_fold,
        "num_workers": args.num_workers,
        "run_name": args.run_name,
        "mlflow_model_name": args.mlflow_model_name,
        "fsdp": args.fsdp
    }

    try:
        if args.fsdp:
            scaling_config = ScalingConfig(
                num_workers=config.get("num_workers", 2),  
                use_gpu=True, 
                resources_per_worker={"CPU": 8, "GPU": 1},
            )
        else:
            scaling_config = ScalingConfig(
                num_workers=config.get("num_workers", 1),  
                use_gpu=True, 
                resources_per_worker={"CPU": 8, "GPU": 1}  # 
            )
        
        run_config = RunConfig(storage_path="s3://ray")
        
        trainer = TorchTrainer(
            train_func,
            train_loop_config=config,
            scaling_config=scaling_config,
            run_config=run_config
        )
        
        result = trainer.fit()
        print("Training completed successfully.")
    finally:
        # ray.shutdown()
        pass