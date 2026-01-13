"""单元测试"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch


class TestUtils:
    """测试工具函数"""
    
    def test_load_config(self):
        from src.utils import load_config
        
        config = load_config("config/default.yaml")
        
        assert "model" in config
        assert "devices" in config
        assert config["model"]["name"] == "Qwen/Qwen1.5-MoE-A2.7B"
    
    def test_get_dtype(self):
        from src.utils import get_dtype
        
        assert get_dtype("float16") == torch.float16
        assert get_dtype("fp16") == torch.float16
        assert get_dtype("bfloat16") == torch.bfloat16
        assert get_dtype("float32") == torch.float32
    
    def test_set_seed(self):
        from src.utils import set_seed
        
        set_seed(42)
        a = torch.rand(10)
        
        set_seed(42)
        b = torch.rand(10)
        
        assert torch.allclose(a, b)


class TestCommunicator:
    """测试通信器"""
    
    def test_stats(self):
        from src.communicator import CommunicationStats
        
        stats = CommunicationStats()
        tensor = torch.randn(100, 100)
        
        stats.add_transfer(tensor, 10.0)
        stats.add_transfer(tensor, 20.0)
        
        assert stats.transfer_count == 2
        assert stats.avg_time_ms == 15.0
    
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2,
        reason="Requires 2 GPUs"
    )
    def test_transfer(self):
        from src.communicator import DeviceCommunicator
        
        comm = DeviceCommunicator(
            attn_device=torch.device("cuda:0"),
            expert_device=torch.device("cuda:1"),
        )
        
        tensor = torch.randn(10, 10, device="cuda:0")
        
        # Transfer to expert device
        result = comm.attn_to_expert(tensor)
        assert result.device == torch.device("cuda:1")
        
        # Transfer back
        result = comm.expert_to_attn(result)
        assert result.device == torch.device("cuda:0")


class TestModelSplitter:
    """测试模型拆分器"""
    
    def test_architecture_detection(self):
        from src.model_splitter import ModelSplitter
        from transformers import AutoConfig
        
        # 这个测试需要网络
        splitter = ModelSplitter("Qwen/Qwen1.5-MoE-A2.7B")
        
        config = AutoConfig.from_pretrained(
            "Qwen/Qwen1.5-MoE-A2.7B",
            trust_remote_code=True,
        )
        
        arch = splitter._detect_architecture(config)
        assert arch == "qwen"


# 如果直接运行此文件
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
