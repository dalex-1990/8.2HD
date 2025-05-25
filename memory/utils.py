import os
import torch
import random
import numpy as np

# Set CUDA environment variables
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        # Print CUDA information
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.device_count() > 0:
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
        
        # Set CUDA device
        torch.cuda.set_device(0)
        
        # Test CUDA with a simple operation using CPU tensor first
        x = torch.zeros(1)
        y = torch.ones(1)
        x = x.cuda()
        y = y.cuda()
        z = x + y  # This should work on any CUDA device
        
        _DEVICE = "cuda"
        print("CUDA initialization successful")
        
    except RuntimeError as e:
        print("\nCUDA initialization failed. Please check:")
        print("1. CUDA is properly installed")
        print("2. NVIDIA drivers are up to date")
        print("3. PyTorch is installed with CUDA support")
        print("4. GPU is not being used by another process")
        print(f"\nDetailed error: {e}")
        raise RuntimeError(f"Failed to initialize CUDA: {e}")
    
    return _DEVICE

def device():
    return _init_device()
