"""模型拆分器：将 MoE 模型的 Attention 和 Expert 层分离到不同设备"""

import logging
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig

from .utils import get_dtype, get_device

logger = logging.getLogger(__name__)


class ModelSplitter:
    """
    将 MoE 模型拆分为 Attention 部分和 Expert 部分
    
    支持的模型架构:
    - Qwen MoE (Qwen1.5-MoE, Qwen2-MoE)
    - Mixtral
    - DeepSeek MoE
    """
    
    # 不同模型架构的层名映射
    ARCHITECTURE_MAPPING = {
        "qwen": {
            "layers": "model.layers",
            "attention": "self_attn",
            "moe": "mlp",  # Qwen MoE 的 mlp 就是 MoE 层
            "input_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "embed": "model.embed_tokens",
            "final_norm": "model.norm",
            "lm_head": "lm_head",
        },
        "mixtral": {
            "layers": "model.layers",
            "attention": "self_attn",
            "moe": "block_sparse_moe",
            "input_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "embed": "model.embed_tokens",
            "final_norm": "model.norm",
            "lm_head": "lm_head",
        },
        "deepseek": {
            "layers": "model.layers",
            "attention": "self_attn",
            "moe": "mlp",  # DeepSeek MoE
            "input_norm": "input_layernorm",
            "post_norm": "post_attention_layernorm",
            "embed": "model.embed_tokens",
            "final_norm": "model.norm",
            "lm_head": "lm_head",
        },
    }
    
    def __init__(
        self,
        model_name: str,
        attn_device: str = "cuda:0",
        expert_device: str = "cuda:1",
        dtype: str = "float16",
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self.attn_device = get_device(attn_device)
        self.expert_device = get_device(expert_device)
        self.dtype = get_dtype(dtype)
        self.trust_remote_code = trust_remote_code
        
        self.model = None
        self.arch_type = None
        self.arch_config = None
        
    def _detect_architecture(self, config) -> str:
        """检测模型架构类型"""
        model_type = config.model_type.lower()
        
        if "qwen" in model_type:
            return "qwen"
        elif "mixtral" in model_type:
            return "mixtral"
        elif "deepseek" in model_type:
            return "deepseek"
        else:
            logger.warning(f"Unknown model type: {model_type}, defaulting to mixtral")
            return "mixtral"
    
    def _get_layers(self, model) -> nn.ModuleList:
        """获取模型的层列表"""
        parts = self.arch_config["layers"].split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part)
        return obj
    
    def _get_module(self, model, path: str):
        """根据路径获取模块"""
        parts = path.split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part)
        return obj
    
    def load_model(self) -> nn.Module:
        """加载模型"""
        logger.info(f"Loading model: {self.model_name}")
        
        config = AutoConfig.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        
        self.arch_type = self._detect_architecture(config)
        self.arch_config = self.ARCHITECTURE_MAPPING[self.arch_type]
        logger.info(f"Detected architecture: {self.arch_type}")
        
        # 先加载到 CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            device_map="cpu",  # 先全部加载到 CPU
        )
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
        return self.model
    
    def split(self) -> nn.Module:
        """
        执行模型拆分
        
        Attention 相关层 -> attn_device
        Expert/MoE 相关层 -> expert_device
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"Splitting model: Attention -> {self.attn_device}, Expert -> {self.expert_device}")
        
        # 1. Embedding 层 -> Attention 设备
        embed = self._get_module(self.model, self.arch_config["embed"])
        embed.to(self.attn_device)
        logger.debug("Moved embedding to attention device")
        
        # 2. 遍历每一层，拆分 Attention 和 MoE
        layers = self._get_layers(self.model)
        
        for i, layer in enumerate(layers):
            # Attention 部分 -> attn_device
            attn = getattr(layer, self.arch_config["attention"])
            input_norm = getattr(layer, self.arch_config["input_norm"])
            
            attn.to(self.attn_device)
            input_norm.to(self.attn_device)
            
            # MoE/Expert 部分 -> expert_device
            moe = getattr(layer, self.arch_config["moe"])
            post_norm = getattr(layer, self.arch_config["post_norm"])
            
            moe.to(self.expert_device)
            post_norm.to(self.expert_device)
            
            if i == 0 or i == len(layers) - 1:
                logger.debug(f"Layer {i}: Attention -> {self.attn_device}, MoE -> {self.expert_device}")
        
        # 3. Final norm 和 LM head -> Expert 设备（因为接在 MoE 后面）
        final_norm = self._get_module(self.model, self.arch_config["final_norm"])
        lm_head = self._get_module(self.model, self.arch_config["lm_head"])
        
        final_norm.to(self.expert_device)
        lm_head.to(self.expert_device)
        
        logger.info("Model split complete!")
        return self.model
    
    def get_device_map(self) -> Dict[str, str]:
        """获取拆分后的设备映射"""
        device_map = {}
        
        if self.model is None:
            return device_map
        
        for name, param in self.model.named_parameters():
            device_map[name] = str(param.device)
        
        return device_map
    
    def print_split_summary(self):
        """打印拆分摘要"""
        if self.model is None:
            print("Model not loaded yet.")
            return
        
        attn_params = 0
        expert_params = 0
        other_params = 0
        
        for name, param in self.model.named_parameters():
            n = param.numel()
            device = str(param.device)
            
            if str(self.attn_device) in device:
                attn_params += n
            elif str(self.expert_device) in device:
                expert_params += n
            else:
                other_params += n
        
        total = attn_params + expert_params + other_params
        
        print("\n" + "=" * 60)
        print("Model Split Summary")
        print("=" * 60)
        print(f"  Attention device ({self.attn_device}):")
        print(f"    Parameters: {attn_params:,} ({attn_params/total*100:.1f}%)")
        print(f"    Memory: ~{attn_params * 2 / 1024**3:.2f} GB (FP16)")
        print(f"  Expert device ({self.expert_device}):")
        print(f"    Parameters: {expert_params:,} ({expert_params/total*100:.1f}%)")
        print(f"    Memory: ~{expert_params * 2 / 1024**3:.2f} GB (FP16)")
        if other_params > 0:
            print(f"  Other:")
            print(f"    Parameters: {other_params:,} ({other_params/total*100:.1f}%)")
        print("=" * 60 + "\n")
