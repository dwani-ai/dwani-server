# utils/device_utils.py
import torch
from logging_config import logger

def setup_device():
    """Set up the device (CPU/GPU) and data type for model inference."""
    if torch.cuda.is_available():
        device = "cuda:0"
        logger.info("GPU will be used for inference")
        torch_dtype = torch.bfloat16
    else:
        device = "cpu"
        logger.info("CPU will be used for inference")
        torch_dtype = torch.float32
    
    # Check CUDA availability and version
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None

    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        capability = torch.cuda.get_device_capability(device_idx)
        compute_capability_float = float(f"{capability[0]}.{capability[1]}")
        logger.info(f"CUDA version: {cuda_version}")
        logger.info(f"CUDA Compute Capability: {compute_capability_float}")
    else:
        logger.info("CUDA is not available on this system.")
    
    return device, torch_dtype