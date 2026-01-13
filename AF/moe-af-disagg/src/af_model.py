"""A/F 分离式模型封装"""

import logging
from typing import Optional, Dict, List, Tuple, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig

from .model_splitter import ModelSplitter
from .communicator import DeviceCommunicator
from .utils import Timer

logger = logging.getLogger(__name__)


class AFDisaggregatedModel:
    """
    Attention-FFN 分离式推理模型
    
    核心思想:
    - Attention 层和 MoE/FFN 层分别在不同设备上执行
    - 每层执行完 Attention 后，将中间结果传输到 Expert 设备
    - 执行 MoE 后，将结果传回 Attention 设备（用于下一层或残差连接）
    
    执行流程 (每一层):
    1. Input -> LayerNorm -> Attention (attn_device)
    2. Transfer: attn_device -> expert_device
    3. LayerNorm -> MoE/FFN (expert_device)
    4. Transfer: expert_device -> attn_device
    5. Residual connection, repeat for next layer
    """
    
    def __init__(
        self,
        model_name: str,
        attn_device: str = "cuda:0",
        expert_device: str = "cuda:1",
        dtype: str = "float16",
        trust_remote_code: bool = True,
        profile_communication: bool = True,
    ):
        self.model_name = model_name
        self.attn_device = torch.device(attn_device)
        self.expert_device = torch.device(expert_device)
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        
        # 组件
        self.splitter = ModelSplitter(
            model_name=model_name,
            attn_device=attn_device,
            expert_device=expert_device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.communicator = DeviceCommunicator(
            attn_device=self.attn_device,
            expert_device=self.expert_device,
            profile=profile_communication,
        )
        self.tokenizer = None
        self.model = None
        
        # 时间统计
        self.timing_stats = {
            "attention_time_ms": 0.0,
            "expert_time_ms": 0.0,
            "communication_time_ms": 0.0,
            "total_time_ms": 0.0,
        }
    
    def load(self):
        """加载并拆分模型"""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Loading and splitting model...")
        self.model = self.splitter.load_model()
        self.splitter.split()
        
        self.splitter.print_split_summary()
        
        # 设置为 eval 模式
        self.model.eval()
        
        return self
    
    def _prepare_inputs(
        self, 
        prompt: str,
    ) -> Dict[str, torch.Tensor]:
        """准备模型输入"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        )
        
        # 输入放在 attention 设备上（embedding 在那里）
        inputs = {k: v.to(self.attn_device) for k, v in inputs.items()}
        
        return inputs
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        生成文本
        
        注意: 这里使用 HuggingFace 的原生 generate()
        实际的 A/F 分离在模型 forward 中自动发生（因为权重已经分离）
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = self._prepare_inputs(prompt)
        
        # 重置通信统计
        self.communicator.reset_stats()
        
        with Timer("Total generation") as timer:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )
        
        self.timing_stats["total_time_ms"] = timer.elapsed * 1000
        
        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True,
        )
        
        return generated_text
    
    @torch.no_grad()
    def forward_with_hooks(
        self,
        prompt: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        带钩子的前向传播，用于详细分析每层的执行
        
        Returns:
            (logits, stats_dict)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = self._prepare_inputs(prompt)
        
        layer_stats = []
        
        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                layer_stats.append({
                    "layer": layer_idx,
                    "type": "attention",
                    "device": str(output[0].device) if isinstance(output, tuple) else str(output.device),
                })
            return hook
        
        def make_moe_hook(layer_idx):
            def hook(module, input, output):
                layer_stats.append({
                    "layer": layer_idx,
                    "type": "moe",
                    "device": str(output.device),
                })
            return hook
        
        # 注册钩子
        handles = []
        arch_config = self.splitter.arch_config
        layers = self.splitter._get_layers(self.model)
        
        for i, layer in enumerate(layers):
            attn = getattr(layer, arch_config["attention"])
            moe = getattr(layer, arch_config["moe"])
            
            handles.append(attn.register_forward_hook(make_attn_hook(i)))
            handles.append(moe.register_forward_hook(make_moe_hook(i)))
        
        # 前向传播
        with Timer("Forward pass"):
            outputs = self.model(**inputs)
        
        # 移除钩子
        for h in handles:
            h.remove()
        
        return outputs.logits, {"layer_stats": layer_stats}
    
    def print_timing_stats(self):
        """打印时间统计"""
        print("\n" + "=" * 60)
        print("Timing Statistics")
        print("=" * 60)
        for key, value in self.timing_stats.items():
            print(f"  {key}: {value:.2f} ms")
        print("=" * 60 + "\n")
        
        self.communicator.print_stats()
    
    def verify_output(
        self,
        prompt: str,
        reference_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        验证分离后的输出是否正确
        
        如果没有提供 reference_model，会重新加载一个未分离的模型进行对比
        """
        from transformers import AutoModelForCausalLM
        
        inputs = self._prepare_inputs(prompt)
        
        # 分离模型的输出
        with torch.no_grad():
            split_outputs = self.model(**inputs)
            split_logits = split_outputs.logits
        
        # 加载参考模型
        if reference_model is None:
            logger.info("Loading reference model for verification...")
            reference_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=getattr(torch, self.dtype.replace("float", "float")),
                trust_remote_code=self.trust_remote_code,
                device_map="auto",
            )
            reference_model.eval()
        
        # 参考模型的输出
        ref_inputs = {k: v.to(reference_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            ref_outputs = reference_model(**ref_inputs)
            ref_logits = ref_outputs.logits
        
        # 比较
        split_logits_cpu = split_logits.float().cpu()
        ref_logits_cpu = ref_logits.float().cpu()
        
        # 计算差异
        max_diff = (split_logits_cpu - ref_logits_cpu).abs().max().item()
        mean_diff = (split_logits_cpu - ref_logits_cpu).abs().mean().item()
        
        # 检查是否接近
        is_close = torch.allclose(
            split_logits_cpu, 
            ref_logits_cpu, 
            rtol=1e-3, 
            atol=1e-3,
        )
        
        result = {
            "is_close": is_close,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "split_shape": tuple(split_logits.shape),
            "ref_shape": tuple(ref_logits.shape),
        }
        
        print("\n" + "=" * 60)
        print("Output Verification")
        print("=" * 60)
        print(f"  Shapes match: {result['split_shape'] == result['ref_shape']}")
        print(f"  Max difference: {result['max_diff']:.6f}")
        print(f"  Mean difference: {result['mean_diff']:.6f}")
        print(f"  Outputs close (rtol=1e-3): {result['is_close']}")
        print("=" * 60 + "\n")
        
        return result


def create_model(
    model_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    attn_device: str = "cuda:0",
    expert_device: str = "cuda:1",
    **kwargs,
) -> AFDisaggregatedModel:
    """
    工厂函数: 创建并加载分离式模型
    
    Args:
        model_name: HuggingFace 模型名称
        attn_device: Attention 层设备
        expert_device: Expert/MoE 层设备
        
    Returns:
        加载完成的 AFDisaggregatedModel
    """
    model = AFDisaggregatedModel(
        model_name=model_name,
        attn_device=attn_device,
        expert_device=expert_device,
        **kwargs,
    )
    model.load()
    return model
