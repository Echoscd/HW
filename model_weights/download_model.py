#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
从 HuggingFace 下载模型权重到本地目录。

用法示例:
    python download_model.py \
        --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --output-dir ./model_weights

    python download_model.py \
        --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
        --output-dir ./model_weights \
        --revision main
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from tqdm import tqdm


def download_model(
    model_name: str,
    output_dir: str,
    revision: str = "main",
    token: str = None,
    resume_download: bool = True,
    local_files_only: bool = False,
    ignore_patterns: list = None,
):
    """
    从 HuggingFace 下载模型权重。

    Args:
        model_name: HuggingFace 模型名称，例如 "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        output_dir: 输出目录
        revision: 模型版本/分支（默认: main）
        token: HuggingFace token（可选，用于私有模型）
        resume_download: 是否支持断点续传（默认: True）
        local_files_only: 是否只使用本地文件（默认: False）
        ignore_patterns: 要忽略的文件模式列表（可选）
    """
    print(f"正在从 HuggingFace 下载模型: {model_name}")
    print(f"  版本: {revision}")
    print(f"  输出目录: {output_dir}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 下载模型
        model_path = snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=None,  # 不使用缓存，直接下载到指定目录
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            token=token,
            resume_download=resume_download,
            local_files_only=local_files_only,
            ignore_patterns=ignore_patterns,
        )

        print(f"\n✓ 模型已成功下载到: {output_path.absolute()}")
        
        # 计算下载的文件大小
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(output_path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        print(f"  文件数量: {file_count}")
        print(f"  总大小: {total_size / 1024 / 1024 / 1024:.2f} GB")
        
        return str(output_path.absolute())

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n提示:")
        print("1. 检查模型名称是否正确")
        print("2. 检查网络连接")
        print("3. 如果是私有模型，请使用 --token 参数提供 HuggingFace token")
        print("4. 如果下载中断，可以重新运行脚本（支持断点续传）")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载模型权重到本地目录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载公开模型
  python download_model.py \\
      --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
      --output-dir ./model_weights/DeepSeek-R1-Distill-Qwen-7B

  # 下载指定版本
  python download_model.py \\
      --model-name "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \\
      --output-dir ./model_weights/DeepSeek-R1-Distill-Qwen-7B \\
      --revision main

  # 下载私有模型（需要 token）
  python download_model.py \\
      --model-name "username/private-model" \\
      --output-dir ./model_weights/private-model \\
      --token YOUR_HF_TOKEN
        """,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace 模型名称，例如 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录路径",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="模型版本/分支（默认: main）",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token（用于私有模型，可选）",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="禁用断点续传",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="只使用本地文件（不下载）",
    )

    args = parser.parse_args()

    # 如果没有提供 token，尝试从环境变量获取
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    download_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        revision=args.revision,
        token=token,
        resume_download=not args.no_resume,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()


