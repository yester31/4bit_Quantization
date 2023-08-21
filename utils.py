import os
import torch
import numpy as np
import random

def genDir(dir_path):
    if not os.path.exists(dir_path):  # if dir_path is not exist
        os.makedirs(dir_path)  # generate directory
        print(f"The directory ({dir_path}) has generated.")
    else:
        print(f"Already {dir_path} exists")


def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def device_check():
    print("pytorch:", torch.__version__)
    print("[Device Check]")
    if torch.cuda.is_available():
        print(f"Torch gpu available : {torch.cuda.is_available()}")
        print(f"The number of gpu device : {torch.cuda.device_count()}")
        for g_idx in range(torch.cuda.device_count()):
            print(f"{g_idx} device name : {torch.cuda.get_device_name(g_idx)}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu:0")
    print(f"device : {device} is available")
    return device