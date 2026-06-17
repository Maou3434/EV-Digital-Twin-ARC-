import os
import logging
import time
import torch

# Try to import psutil for CPU/RAM profiling
try:
    import psutil
except ImportError:
    psutil = None

# Try to import pynvml/nvidia-ml-py for advanced GPU metrics
try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
except Exception:
    HAS_NVML = False

def setup_logger(name, log_file, level=logging.INFO):
    """Sets up a logger that prints to both console and a log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

def get_system_metrics(device=None):
    """
    Returns a dictionary of system metrics including:
    - CPU usage (%)
    - RAM usage (allocated/total, %)
    - GPU memory usage (allocated, reserved, max allocated in MB)
    - NVML metrics if available (GPU utilization, power draw, temperature)
    """
    metrics = {
        "cpu_percent": None,
        "ram_used_gb": None,
        "ram_total_gb": None,
        "ram_percent": None,
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
        "gpu_max_allocated_mb": 0.0,
        "gpu_utilization": None,
        "gpu_temp_c": None,
        "gpu_power_w": None
    }
    
    # 1. CPU & RAM metrics
    if psutil is not None:
        try:
            metrics["cpu_percent"] = psutil.cpu_percent()
            virtual_mem = psutil.virtual_memory()
            metrics["ram_used_gb"] = virtual_mem.used / (1024 ** 3)
            metrics["ram_total_gb"] = virtual_mem.total / (1024 ** 3)
            metrics["ram_percent"] = virtual_mem.percent
        except Exception:
            pass
            
    # 2. PyTorch GPU Memory metrics
    if torch.cuda.is_available():
        try:
            current_device = device if device is not None else torch.cuda.current_device()
            metrics["gpu_allocated_mb"] = torch.cuda.memory_allocated(current_device) / (1024 ** 2)
            metrics["gpu_reserved_mb"] = torch.cuda.memory_reserved(current_device) / (1024 ** 2)
            metrics["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated(current_device) / (1024 ** 2)
        except Exception:
            pass
            
    # 3. NVML NVidia GPU metrics (Utilization, Power, Temp)
    if HAS_NVML:
        try:
            # Assumes device index maps to NVML device index (usually matches)
            device_idx = device.index if isinstance(device, torch.device) and device.index is not None else 0
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            
            # Utilization (GPU and memory controllers)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["gpu_utilization"] = util.gpu
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            metrics["gpu_temp_c"] = temp
            
            # Power draw in Watts (NVML reports in milliwatts)
            power = pynvml.nvmlDeviceGetPowerUsage(handle)
            metrics["gpu_power_w"] = power / 1000.0
        except Exception:
            pass
            
    return metrics

def log_system_metrics(logger, prefix="System Metrics"):
    """Convenience helper to log metrics to the logger."""
    metrics = get_system_metrics()
    
    gpu_str = ""
    if torch.cuda.is_available():
        gpu_str = (f" | GPU Mem Alloc: {metrics['gpu_allocated_mb']:.1f}MB, "
                   f"Res: {metrics['gpu_reserved_mb']:.1f}MB, "
                   f"Max: {metrics['gpu_max_allocated_mb']:.1f}MB")
        if metrics["gpu_utilization"] is not None:
            gpu_str += f" | GPU Util: {metrics['gpu_utilization']}%"
        if metrics["gpu_temp_c"] is not None:
            gpu_str += f" | GPU Temp: {metrics['gpu_temp_c']}°C"
        if metrics["gpu_power_w"] is not None:
            gpu_str += f" | GPU Power: {metrics['gpu_power_w']:.1f}W"
            
    cpu_ram_str = ""
    if metrics["cpu_percent"] is not None:
        cpu_ram_str = (f" | CPU: {metrics['cpu_percent']:.1f}%"
                       f" | RAM: {metrics['ram_used_gb']:.1f}/{metrics['ram_total_gb']:.1f}GB ({metrics['ram_percent']:.1f}%)")
                       
    logger.info(f"{prefix}{cpu_ram_str}{gpu_str}")
