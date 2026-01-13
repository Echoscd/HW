"""工具函数"""

import os
import yaml
import random
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(level: str = "INFO") -> logging.Logger:
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(dtype_str: str) -> torch.dtype:
    """字符串转 torch dtype"""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype_str.lower(), torch.float16)


def get_device(device_str: str) -> torch.device:
    """字符串转 torch device"""
    return torch.device(device_str)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "total_mb": total * 2 / 1024 / 1024,  # 假设 fp16
    }


def print_gpu_memory():
    """打印 GPU 显存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")


def get_layer_device_map(model) -> Dict[str, str]:
    """获取模型各层所在设备"""
    device_map = {}
    for name, param in model.named_parameters():
        device_map[name] = str(param.device)
    return device_map


def print_layer_devices(model, max_layers: int = 5):
    """打印模型各层设备分布（简化版）"""
    print("\n" + "=" * 50)
    print("Layer Device Distribution:")
    print("=" * 50)
    
    device_map = get_layer_device_map(model)
    shown = 0
    
    for name, device in device_map.items():
        if shown < max_layers:
            print(f"  {name[:60]:<60} -> {device}")
            shown += 1
        elif shown == max_layers:
            print(f"  ... (and {len(device_map) - max_layers} more layers)")
            break
    
    print("=" * 50 + "\n")


class Timer:
    """简单的计时器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available():
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            self.elapsed = self.start_time.elapsed_time(end_event) / 1000  # ms -> s
        else:
            import time
            self.elapsed = time.time() - self.start_time
        
        if self.name:
            print(f"[Timer] {self.name}: {self.elapsed*1000:.2f} ms")
