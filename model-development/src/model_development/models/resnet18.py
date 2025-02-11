

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from model_development.constants import ARTIFACTS_DIR
from model_development.utils import logger

from pathlib import Path
import os

def get_latest_checkpoint(model_name, checkpoint_dir):
    # checkpoint_files = sorted(checkpoint_dir.glob("*.pth"))
    # if not checkpoint_files:
    #     raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    # return checkpoint_files[-1]
    return "dummy_checkpoint.pth"

def load_resnet18_torch(num_classes, weights="pretrained", mode="inference", device="mps"):
    """
    Load a ResNet18 model with either official pretrained weights or a local checkpoint.
    The function sets the trainability of layers based on the desired mode, replaces
    the final classification layer for a dataset with 2 to 10 classes (if needed),
    and moves the model to the specified device.
    
    Args:
        num_classes (int): The number of classes for the final classification layer.
                           Must be between 2 and 10.
        weights (str): Which weights to load. Options are:
            - "pretrained": use the official torchvision pretrained weights.
            - "local": load from the latest local checkpoint.
        mode (str): Mode in which to load the model. Options:
            - "inference": Freeze all layers (for inference).
            - "finetune": Freeze all layers except the final fully connected layer.
            - "train": Unfreeze all layers (for training from scratch or full fine-tuning).
        device (str or torch.device): The device on which to load the model (default: "cpu").
            
    Returns:
        torch.nn.Module: The configured ResNet18 model.
    """
    
    if weights == "pretrained":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif weights == "local":
        latest_checkpoint = get_latest_checkpoint("resnet18", ARTIFACTS_DIR / "checkpoints")
        model = models.resnet18()
        state_dict = torch.load(latest_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Invalid weights option. Use 'pretrained' or 'local'.")

    # Adjust parameter freezing based on the desired mode.
    if mode == "inference":
        for param in model.parameters():
            param.requires_grad = False
    elif mode == "finetune":
        for name, param in model.named_parameters():
            param.requires_grad = ("fc" in name)
    elif mode == "train":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid mode. Choose 'inference', 'finetune', or 'train'.")

    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)
    
    return model


def get_hyperparameters(model, args):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    return criterion, optimizer, scheduler

def resume_from_checkpoint(model, args):
    """
    Initializes optimizer and scheduler, and resumes training from a checkpoint if available.
    
    Args:
        model (torch.nn.Module): The model whose parameters will be optimized.
        args (Namespace): The argument namespace containing hyperparameters.
        
    Returns:
        tuple: (criterion, optimizer, scheduler, start_epoch)
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info("Resumed from checkpoint: %s at epoch %d", args.checkpoint, start_epoch)

    return criterion, optimizer, scheduler, start_epoch


