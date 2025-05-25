import os
import torch
import random
import numpy as np


def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def device(force_cpu=False):
    if force_cpu:
        return "cpu"
    try:
        if torch.cuda.is_available():
            # Test CUDA with a simple operation
            x = torch.tensor([1.0]).cuda()
            return "cuda"
    except RuntimeError:
        print("CUDA is available but encountered an error. Falling back to CPU.")
    return "cpu"
