import os
import torch
import random
import numpy as np

# Global device variable
_DEVICE = None

def set_seed(env, seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def _init_device():
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
        
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This code requires GPU to run.")
        
    try:
        # Initialize CUDA
        torch.cuda.init()
        # Test CUDA with a simple operation
        x = torch.tensor([1.0], device="cuda")
        _DEVICE = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to initialize CUDA: {e}")
    
    return _DEVICE

def device():
    return _init_device()
