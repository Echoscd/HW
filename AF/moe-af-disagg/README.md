# MoE Attention-FFN Disaggregation

基于 HuggingFace Transformers 的轻量级 MoE 模型 Attention-FFN 分离推理框架。

## 目标

将 MoE 模型的 Attention 层和 Expert (FFN) 层分离到不同的 GPU 上执行，验证分离式推理的可行性。

```
原始执行:
┌─────────────────────────────────┐
│  GPU 0: Attn -> MoE -> Attn -> MoE ...  │
└─────────────────────────────────┘

分离后:
┌─────────────────┐     ┌─────────────────┐
│  GPU 0: Attn    │ <-> │  GPU 1: MoE     │
└─────────────────┘     └─────────────────┘
```

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 单卡测试（用 CPU 模拟第二设备）
python scripts/run_inference.py --model Qwen/Qwen1.5-MoE-A2.7B --mode single

# 双卡分离
python scripts/run_inference.py --model Qwen/Qwen1.5-MoE-A2.7B --mode split

# 正确性验证
python scripts/verify_correctness.py --model Qwen/Qwen1.5-MoE-A2.7B
```

## 项目结构

```
moe-af-disagg/
├── src/
│   ├── model_splitter.py    # 模型拆分逻辑
│   ├── af_model.py          # 分离式模型封装
│   ├── communicator.py      # GPU 间通信
│   └── utils.py             # 工具函数
├── config/
│   └── default.yaml         # 配置文件
├── scripts/
│   ├── run_inference.py     # 推理入口
│   └── verify_correctness.py # 正确性验证
└── tests/
    └── test_split.py        # 单元测试
```

## 支持的模型

- [x] Qwen/Qwen1.5-MoE-A2.7B (推荐，小巧)
- [x] mistralai/Mixtral-8x7B-v0.1
- [x] deepseek-ai/deepseek-moe-16b-base

## 配置说明

编辑 `config/default.yaml`:

```yaml
model:
  name: "Qwen/Qwen1.5-MoE-A2.7B"
  dtype: "float16"

devices:
  attention: "cuda:0"    # Attention 层所在设备
  expert: "cuda:1"       # Expert 层所在设备
  # expert: "cpu"        # 单卡时用 CPU 模拟
```

## 性能说明

本框架仅用于验证分离式推理的可行性，**不追求生产级性能**。

如需高性能方案，请参考：
- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [MegaScale-Infer](https://arxiv.org/abs/2504.02263)

## License

MIT
