"""设备间通信器"""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class CommunicationStats:
    """通信统计"""
    transfer_count: int = 0
    total_bytes: int = 0
    total_time_ms: float = 0.0
    
    def add_transfer(self, tensor: torch.Tensor, time_ms: float):
        self.transfer_count += 1
        self.total_bytes += tensor.numel() * tensor.element_size()
        self.total_time_ms += time_ms
    
    @property
    def avg_time_ms(self) -> float:
        if self.transfer_count == 0:
            return 0.0
        return self.total_time_ms / self.transfer_count
    
    @property
    def throughput_gbps(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return (self.total_bytes * 8) / (self.total_time_ms / 1000) / 1e9
    
    def __str__(self):
        return (
            f"Transfers: {self.transfer_count}, "
            f"Total: {self.total_bytes / 1024**2:.2f} MB, "
            f"Avg time: {self.avg_time_ms:.2f} ms, "
            f"Throughput: {self.throughput_gbps:.2f} Gbps"
        )


class DeviceCommunicator:
    """
    处理不同设备之间的张量传输
    
    目前实现: 简单的 .to(device) 同步传输
    未来可扩展: NCCL, RDMA, 异步传输
    """
    
    def __init__(
        self,
        attn_device: torch.device,
        expert_device: torch.device,
        async_transfer: bool = False,
        profile: bool = True,
    ):
        self.attn_device = attn_device
        self.expert_device = expert_device
        self.async_transfer = async_transfer
        self.profile = profile
        
        # 通信统计
        self.stats_attn_to_expert = CommunicationStats()
        self.stats_expert_to_attn = CommunicationStats()
        
        # CUDA streams for async transfer
        self._transfer_stream = None
        if async_transfer and torch.cuda.is_available():
            self._transfer_stream = torch.cuda.Stream()
    
    def _sync_if_needed(self):
        """同步 CUDA 流"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def _measure_transfer(self, tensor: torch.Tensor, dst_device: torch.device) -> tuple:
        """测量传输时间并返回 (传输后的张量, 耗时ms)"""
        if not self.profile:
            return tensor.to(dst_device), 0.0
        
        self._sync_if_needed()
        
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            result = tensor.to(dst_device)
            end_event.record()
            
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            start = time.time()
            result = tensor.to(dst_device)
            elapsed_ms = (time.time() - start) * 1000
        
        return result, elapsed_ms
    
    def attn_to_expert(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        从 Attention 设备传输到 Expert 设备
        
        Args:
            tensor: Attention 层的输出张量
            
        Returns:
            在 Expert 设备上的张量
        """
        if tensor.device == self.expert_device:
            return tensor
        
        result, elapsed_ms = self._measure_transfer(tensor, self.expert_device)
        
        if self.profile:
            self.stats_attn_to_expert.add_transfer(tensor, elapsed_ms)
        
        return result
    
    def expert_to_attn(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        从 Expert 设备传输到 Attention 设备
        
        Args:
            tensor: Expert/MoE 层的输出张量
            
        Returns:
            在 Attention 设备上的张量
        """
        if tensor.device == self.attn_device:
            return tensor
        
        result, elapsed_ms = self._measure_transfer(tensor, self.attn_device)
        
        if self.profile:
            self.stats_expert_to_attn.add_transfer(tensor, elapsed_ms)
        
        return result
    
    def print_stats(self):
        """打印通信统计"""
        print("\n" + "=" * 60)
        print("Communication Statistics")
        print("=" * 60)
        print(f"  Attn -> Expert: {self.stats_attn_to_expert}")
        print(f"  Expert -> Attn: {self.stats_expert_to_attn}")
        
        total_time = (
            self.stats_attn_to_expert.total_time_ms + 
            self.stats_expert_to_attn.total_time_ms
        )
        print(f"  Total communication time: {total_time:.2f} ms")
        print("=" * 60 + "\n")
    
    def reset_stats(self):
        """重置统计"""
        self.stats_attn_to_expert = CommunicationStats()
        self.stats_expert_to_attn = CommunicationStats()
