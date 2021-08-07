import random
import torch
import os
import numpy as np
import config
import copy

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print(" Saving Checkpoint: ")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print(" Loading Checkpoint: ")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it'll jsut have learning rate of old checkpoint and it'll lead to many hours of debugging:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# Not really useful here:
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


