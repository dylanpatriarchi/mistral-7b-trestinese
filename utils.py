"""
Utility functions for Mistral-7B Trestinese fine-tuning.
"""

import os
import yaml
import logging
import random
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, log_level.upper()),
    )
    logger = logging.getLogger(__name__)
    return logger


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_prompt(instruction: str, input_text: str = "", output_text: str = "") -> str:
    """
    Create a formatted prompt for the model.
    
    Args:
        instruction: Task instruction
        input_text: Input text (Italian)
        output_text: Expected output text (Trestinese) - used during training
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Output:
{output_text}"""
    
    return prompt


def create_inference_prompt(italian_text: str) -> str:
    """
    Create a prompt for inference (without output).
    
    Args:
        italian_text: Italian text to translate
        
    Returns:
        Formatted prompt for inference
    """
    instruction = "Translate the following Italian text to Trestinese dialect."
    prompt = f"""### Instruction:
{instruction}

### Input:
{italian_text}

### Output:
"""
    return prompt


def extract_output_from_response(response: str) -> str:
    """
    Extract the output section from a model response.
    
    Args:
        response: Full model response
        
    Returns:
        Extracted output text
    """
    # Look for the Output section
    if "### Output:" in response:
        output = response.split("### Output:")[-1].strip()
        # Remove any trailing instruction/input sections
        if "### Instruction:" in output:
            output = output.split("### Instruction:")[0].strip()
        return output
    return response.strip()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0
    }


def format_size(num_bytes: int) -> str:
    """
    Format bytes to human-readable size.
    
    Args:
        num_bytes: Number of bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def get_model_size(model: torch.nn.Module) -> str:
    """
    Calculate the size of a model in memory.
    
    Args:
        model: PyTorch model
        
    Returns:
        Formatted model size string
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return format_size(total_size)


def print_model_info(model: torch.nn.Module, logger: Optional[logging.Logger] = None) -> None:
    """
    Print detailed information about the model.
    
    Args:
        model: PyTorch model
        logger: Optional logger instance
    """
    param_info = count_parameters(model)
    model_size = get_model_size(model)
    
    info_message = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║                    MODEL INFORMATION                     ║
    ╠══════════════════════════════════════════════════════════╣
    ║ Total Parameters:     {param_info['total_params']:>30,} ║
    ║ Trainable Parameters: {param_info['trainable_params']:>30,} ║
    ║ Trainable Percentage: {param_info['trainable_percentage']:>29.2f}% ║
    ║ Model Size:           {model_size:>30} ║
    ╚══════════════════════════════════════════════════════════╝
    """
    
    if logger:
        logger.info(info_message)
    else:
        print(info_message)


def ensure_dir(directory: str) -> None:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device instance
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_gpu_utilization() -> None:
    """
    Print current GPU memory utilization.
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {format_size(torch.cuda.memory_allocated(i))}")
            print(f"  Memory Reserved:  {format_size(torch.cuda.memory_reserved(i))}")

