import time 
import torch 
from typing import Union
from pathlib import Path
import random 
import numpy as np
import os

def save_checkpoint(epoch, model, optimizer, model_kwargs, filename: Union[str, Path]):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_kwargs": model_kwargs,
    }
    time.sleep(3)
    torch.save(state, filename)


def set_seed_all(seed: int = 0):
    if not isinstance(seed, int):
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.environ["PYTHONHASHSEED"] = str(seed)
    return None