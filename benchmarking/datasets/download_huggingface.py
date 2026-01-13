#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
从 HuggingFace 下载数据集并保存到本地文件。

用法示例:
    python download_huggingface.py \
        --dataset-path "tatsu-lab/alpaca" \
        --output-dir ./datasets \
        --split train \
        --num-samples 1000

    python download_huggingface.py \
        --dataset-path "HuggingFaceH4/ultrafeedback_binarized" \
        --subset default \
        --split train_sft \
        --output-format jsonl \
        --output-dir ./datasets
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm


def download_dataset(
    dataset_path: str,
    output_dir: str,
    split: str = "train",
    subset: Optional[str] = None,
    num_samples: Optional[int] = None,
    output_format: str = "jsonl",
    output_filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """
    从 HuggingFace 下载数据集并保存到本地文件。

    Args:
        dataset_path: HuggingFace 数据集路径，例如 "tatsu-lab/alpaca"
        output_dir: 输出目录
        split: 数据集分割，例如 "train", "test", "validation"
        subset: 数据集子集名称（可选）
        num_samples: 要下载的样本数量（可选，None 表示下载全部）
        output_format: 输出格式，"jsonl" 或 "json"
        output_filename: 输出文件名（可选，默认自动生成）
        cache_dir: HuggingFace 缓存目录（可选）
        trust_remote_code: 是否信任远程代码
    """
    print(f"正在从 HuggingFace 下载数据集: {dataset_path}")
    if subset:
        print(f"  子集: {subset}")
    print(f"  分割: {split}")
    if num_samples:
        print(f"  样本数量: {num_samples}")

    # 加载数据集
    try:
        dataset = load_dataset(
            dataset_path,
            name=subset,
            split=split,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
    except Exception as e:
        print(f"错误: 无法加载数据集: {e}")
        print("\n提示:")
        print("1. 检查数据集路径是否正确")
        print("2. 检查是否需要指定 --subset 参数")
        print("3. 检查是否需要 --trust-remote-code 参数")
        raise

    # 获取数据集信息
    total_samples = len(dataset)
    print(f"数据集总样本数: {total_samples}")

    # 限制样本数量
    if num_samples and num_samples < total_samples:
        dataset = dataset.select(range(num_samples))
        print(f"已限制为前 {num_samples} 个样本")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成输出文件名
    if output_filename is None:
        # 从 dataset_path 提取名称
        dataset_name = dataset_path.replace("/", "_")
        if subset:
            dataset_name += f"_{subset}"
        dataset_name += f"_{split}"
        if num_samples:
            dataset_name += f"_{num_samples}samples"
        output_filename = f"{dataset_name}.{output_format}"
    else:
        # 确保文件名包含正确的扩展名
        if not output_filename.endswith(f".{output_format}"):
            output_filename = f"{output_filename}.{output_format}"

    output_file = output_path / output_filename
    print(f"输出文件: {output_file}")

    # 保存数据集
    print("正在保存数据集...")
    if output_format == "jsonl":
        # 保存为 JSONL 格式（每行一个 JSON 对象）
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc="保存进度", total=len(dataset)):
                json_str = json.dumps(item, ensure_ascii=False)
                f.write(json_str + "\n")
    elif output_format == "json":
        # 保存为 JSON 格式（单个 JSON 数组）
        data_list = []
        for item in tqdm(dataset, desc="保存进度", total=len(dataset)):
            data_list.append(item)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"不支持的输出格式: {output_format}，支持 'jsonl' 或 'json'")

    print(f"\n✓ 数据集已成功保存到: {output_file}")
    print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  样本数量: {len(dataset)}")


def list_dataset_info(dataset_path: str, subset: Optional[str] = None):
    """
    列出数据集的信息，包括可用的 splits 和配置。

    Args:
        dataset_path: HuggingFace 数据集路径
        subset: 数据集子集名称（可选）
    """
    print(f"正在获取数据集信息: {dataset_path}")
    if subset:
        print(f"  子集: {subset}")

    try:
        # 获取数据集信息
        from datasets import get_dataset_config_names, get_dataset_infos

        # 获取所有配置
        if subset is None:
            configs = get_dataset_config_names(dataset_path)
            print(f"\n可用配置: {list(configs)}")
            if configs:
                print(f"使用第一个配置: {configs[0]}")
                subset = configs[0]

        # 获取数据集信息
        infos = get_dataset_infos(dataset_path)
        if subset and subset in infos:
            info = infos[subset]
            print(f"\n数据集信息:")
            print(f"  配置名称: {subset}")
            print(f"  描述: {info.description[:200] if info.description else 'N/A'}")
            print(f"  特征: {list(info.features.keys()) if info.features else 'N/A'}")
            print(f"  分割: {list(info.splits.keys()) if info.splits else 'N/A'}")
            if info.splits:
                for split_name, split_info in info.splits.items():
                    print(f"    - {split_name}: {split_info.num_examples} 样本")
        else:
            print("\n数据集信息:")
            for config_name, info in infos.items():
                print(f"\n配置: {config_name}")
                print(f"  描述: {info.description[:200] if info.description else 'N/A'}")
                if info.splits:
                    for split_name, split_info in info.splits.items():
                        print(f"    - {split_name}: {split_info.num_examples} 样本")

    except Exception as e:
        print(f"无法获取数据集信息: {e}")
        print("尝试直接加载数据集...")
        try:
            dataset = load_dataset(dataset_path, name=subset, split="train")
            print(f"成功加载，样本数: {len(dataset)}")
            if len(dataset) > 0:
                print(f"示例特征: {list(dataset[0].keys())}")
        except Exception as e2:
            print(f"无法加载数据集: {e2}")


def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 下载数据集并保存到本地文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载完整数据集
  python download_huggingface.py --dataset-path "tatsu-lab/alpaca" --output-dir ./datasets

  # 下载指定数量的样本
  python download_huggingface.py \\
      --dataset-path "tatsu-lab/alpaca" \\
      --output-dir ./datasets \\
      --num-samples 1000

  # 下载指定 split 和 subset
  python download_huggingface.py \\
      --dataset-path "HuggingFaceH4/ultrafeedback_binarized" \\
      --subset default \\
      --split train_sft \\
      --output-dir ./datasets

  # 列出数据集信息
  python download_huggingface.py \\
      --dataset-path "tatsu-lab/alpaca" \\
      --list-info
        """,
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="HuggingFace 数据集路径，例如 'tatsu-lab/alpaca'",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="输出目录（默认: ./datasets）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="数据集分割，例如 'train', 'test', 'validation'（默认: train）",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="数据集子集名称（可选）",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="要下载的样本数量（可选，默认下载全部）",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="输出格式: 'jsonl'（每行一个JSON对象）或 'json'（单个JSON数组）（默认: jsonl）",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="输出文件名（可选，默认自动生成）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace 缓存目录（可选）",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="信任远程代码（某些数据集需要此选项）",
    )
    parser.add_argument(
        "--list-info",
        action="store_true",
        help="列出数据集信息而不下载",
    )

    args = parser.parse_args()

    if args.list_info:
        list_dataset_info(args.dataset_path, args.subset)
    else:
        download_dataset(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            split=args.split,
            subset=args.subset,
            num_samples=args.num_samples,
            output_format=args.output_format,
            output_filename=args.output_filename,
            cache_dir=args.cache_dir,
            trust_remote_code=args.trust_remote_code,
        )


if __name__ == "__main__":
    main()
