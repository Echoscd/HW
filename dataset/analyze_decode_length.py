#!/usr/bin/env python3
"""
下载并分析 shareGPT 和 burstGPT 的 decode length 分布
"""

import json
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict, Any, Optional
import argparse

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("警告: datasets 库未安装，无法从 HuggingFace 加载数据集。使用 pip install datasets 安装。")

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    print("警告: tiktoken 库未安装，将使用单词数代替 token 数。使用 pip install tiktoken 安装。")


def download_file(url: str, save_path: str) -> None:
    """下载文件"""
    print(f"正在下载: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\r进度: {percent:.1f}%", end='', flush=True)
    print(f"\n下载完成: {save_path}")


def load_sharegpt_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 shareGPT 数据（支持单个文件或目录），采样前 50000 条"""
    if os.path.isdir(file_path):
        # 如果是目录，查找所有 JSON 文件
        print(f"正在从目录加载 shareGPT 数据: {file_path}")
        json_files = [f for f in os.listdir(file_path) if f.endswith('.json')]
        json_files.sort()
        print(f"找到 {len(json_files)} 个 JSON 文件: {json_files}")
        
        all_data = []
        for json_file in json_files:
            full_path = os.path.join(file_path, json_file)
            print(f"  正在加载: {json_file}")
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
                print(f"    已加载 {len(data)} 条记录，总计 {len(all_data)} 条")
    else:
        # 单个文件
        print(f"正在加载 shareGPT 数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    
    # 采样前 50000 条数据
    sample_size = 50000
    if len(all_data) > sample_size:
        print(f"取前 {sample_size} 条数据（从 {len(all_data)} 条中）...")
        all_data = all_data[:sample_size]
        print(f"采样完成，共 {len(all_data)} 条记录")
    
    return all_data


def load_burstgpt_data(file_path: str) -> List[Dict[str, Any]]:
    """加载 burstGPT 数据"""
    print(f"正在加载 burstGPT 数据: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_burstgpt_from_hf(dataset_name: str = "lzzmm/BurstGPT", split: Optional[str] = None):
    """从 HuggingFace 加载 burstGPT 数据集，返回数据集对象"""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets 库未安装，请使用 pip install datasets 安装")
    
    print(f"正在从 HuggingFace 加载 burstGPT 数据集: {dataset_name}")
    if split:
        ds = load_dataset(dataset_name, split=split)
    else:
        ds = load_dataset(dataset_name)
        # 尝试获取默认的 split
        if hasattr(ds, 'keys'):
            split = list(ds.keys())[0]
            ds = ds[split]
        else:
            ds = ds['train'] if 'train' in ds else list(ds.values())[0]
    
    print(f"数据集 split: {split if split else 'default'}")
    print(f"数据集大小: {len(ds)}")
    
    # 直接取前50000条数据
    sample_size = 50000
    if len(ds) > sample_size:
        print(f"取前 {sample_size} 条数据（从 {len(ds)} 条中）...")
        ds = ds.select(range(sample_size))
        print(f"采样完成，共 {len(ds)} 条记录")
    
    return ds


def load_lmsys_chat_from_hf(dataset_name: str = "lmsys/lmsys-chat-1m", split: Optional[str] = None):
    """从 HuggingFace 加载 lmsys-chat-1m 数据集，返回数据集对象"""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets 库未安装，请使用 pip install datasets 安装")
    
    print(f"正在从 HuggingFace 加载 lmsys-chat-1m 数据集: {dataset_name}")
    try:
        if split:
            ds = load_dataset(dataset_name, split=split)
        else:
            ds = load_dataset(dataset_name)
            # 尝试获取默认的 split
            if hasattr(ds, 'keys'):
                split = list(ds.keys())[0]
                ds = ds[split]
            else:
                ds = ds['train'] if 'train' in ds else list(ds.values())[0]
    except Exception as e:
        error_msg = str(e)
        if 'gated' in error_msg.lower() or 'authenticated' in error_msg.lower():
            print(f"\n错误: 数据集 {dataset_name} 需要认证才能访问。")
            print("请先使用以下命令登录 HuggingFace:")
            print("  huggingface-cli login")
            print("或者设置环境变量:")
            print("  export HF_TOKEN=your_token_here")
            raise Exception(f"需要认证才能访问数据集 {dataset_name}。请先运行 'huggingface-cli login' 登录。") from e
        else:
            raise
    
    print(f"数据集 split: {split if split else 'default'}")
    print(f"数据集大小: {len(ds)}")
    
    # 直接取前50000条数据
    sample_size = 50000
    if len(ds) > sample_size:
        print(f"取前 {sample_size} 条数据（从 {len(ds)} 条中）...")
        ds = ds.select(range(sample_size))
        print(f"采样完成，共 {len(ds)} 条记录")
    
    return ds


def load_wildchat_from_hf(dataset_name: str = "allenai/WildChat-1M", split: Optional[str] = None):
    """从 HuggingFace 加载 WildChat-1M 数据集，返回数据集对象"""
    if not DATASETS_AVAILABLE:
        raise ImportError("datasets 库未安装，请使用 pip install datasets 安装")
    
    print(f"正在从 HuggingFace 加载 WildChat-1M 数据集: {dataset_name}")
    try:
        if split:
            ds = load_dataset(dataset_name, split=split)
        else:
            ds = load_dataset(dataset_name)
            # 尝试获取默认的 split
            if hasattr(ds, 'keys'):
                split = list(ds.keys())[0]
                ds = ds[split]
            else:
                ds = ds['train'] if 'train' in ds else list(ds.values())[0]
    except Exception as e:
        error_msg = str(e)
        if 'gated' in error_msg.lower() or 'authenticated' in error_msg.lower():
            print(f"\n错误: 数据集 {dataset_name} 需要认证才能访问。")
            print("请先使用以下命令登录 HuggingFace:")
            print("  huggingface-cli login")
            print("或者设置环境变量:")
            print("  export HF_TOKEN=your_token_here")
            raise Exception(f"需要认证才能访问数据集 {dataset_name}。请先运行 'huggingface-cli login' 登录。") from e
        else:
            raise
    
    print(f"数据集 split: {split if split else 'default'}")
    print(f"数据集大小: {len(ds)}")
    
    # 直接取前50000条数据
    sample_size = 50000
    if len(ds) > sample_size:
        print(f"取前 {sample_size} 条数据（从 {len(ds)} 条中）...")
        ds = ds.select(range(sample_size))
        print(f"采样完成，共 {len(ds)} 条记录")
    
    return ds


def extract_decode_lengths_lmsys_chat(ds) -> List[int]:
    """从 lmsys-chat-1m 数据集提取 decode length（从 conversation 字段中 role="assistant" 的回答）"""
    total_items = len(ds)
    print(f"正在提取 lmsys-chat-1m decode length，共 {total_items} 条记录...")
    
    # 调试：查看第一条数据的字段
    if total_items > 0:
        first_item = ds[0]
        if isinstance(first_item, dict):
            print(f"  第一条数据的字段: {list(first_item.keys())[:10]}")
            if 'conversation' in first_item:
                conv = first_item['conversation']
                if isinstance(conv, list) and len(conv) > 0:
                    print(f"  conversation 字段示例: {conv[0] if isinstance(conv[0], dict) else type(conv[0])}")
    
    lengths = []
    found_count = 0
    zero_count = 0
    missing_count = 0
    
    for idx in range(total_items):
        if (idx + 1) % 10000 == 0:
            print(f"  处理进度: {idx + 1}/{total_items} ({100*(idx+1)/total_items:.1f}%) - 已找到: {found_count}, 0 token: {zero_count}, 缺失: {missing_count}")
        
        item = ds[idx]
        
        # 从 conversation 字段中提取 role="assistant" 的回答
        if 'conversation' in item:
            conversation = item['conversation']
            if isinstance(conversation, list):
                for msg in conversation:
                    if isinstance(msg, dict):
                        # 查找 role="assistant" 的消息
                        if msg.get('role') == 'assistant':
                            content = msg.get('content', '')
                            if content and isinstance(content, str):
                                # 使用空格分割计算单词数
                                token_count = len(content.split())
                                if token_count > 0:  # 过滤掉 0 token
                                    lengths.append(token_count)
                                    found_count += 1
                                else:
                                    zero_count += 1
                                break  # 只取第一个 assistant 回复
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    print(f"提取完成，共提取 {len(lengths)} 个 decode length")
    print(f"  统计: 有效数据 {found_count}, 0 token 数据 {zero_count}, 缺失数据 {missing_count}")
    return lengths


def extract_decode_lengths_wildchat(ds) -> List[int]:
    """从 WildChat-1M 数据集提取 decode length（从对话列表中 role="assistant" 的回答）"""
    total_items = len(ds)
    print(f"正在提取 WildChat-1M decode length，共 {total_items} 条记录...")
    
    # 调试：查看第一条数据的字段
    if total_items > 0:
        first_item = ds[0]
        if isinstance(first_item, dict):
            print(f"  第一条数据的字段: {list(first_item.keys())[:10]}")
        elif isinstance(first_item, list):
            print(f"  数据格式: 列表，长度 {len(first_item)}")
            if len(first_item) > 0 and isinstance(first_item[0], dict):
                print(f"  列表元素示例字段: {list(first_item[0].keys())[:5]}")
    
    lengths = []
    found_count = 0
    zero_count = 0
    missing_count = 0
    
    for idx in range(total_items):
        if (idx + 1) % 10000 == 0:
            print(f"  处理进度: {idx + 1}/{total_items} ({100*(idx+1)/total_items:.1f}%) - 已找到: {found_count}, 0 token: {zero_count}, 缺失: {missing_count}")
        
        item = ds[idx]
        conversation = None
        
        # 处理不同的数据格式
        if isinstance(item, list):
            # 如果 item 本身就是列表（对话列表）
            conversation = item
        elif isinstance(item, dict):
            # 如果 item 是字典，查找可能的对话字段
            # 尝试常见的字段名
            for key in ['conversation', 'messages', 'chat', 'dialogue', 'data']:
                if key in item:
                    conversation = item[key]
                    break
            # 如果没有找到，检查 item 是否直接包含 role 字段（单条消息）
            if conversation is None and 'role' in item:
                conversation = [item]
        
        # 从对话列表中提取 role="assistant" 的回答
        if conversation and isinstance(conversation, list):
            for msg in conversation:
                if isinstance(msg, dict):
                    # 查找 role="assistant" 的消息
                    if msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if content and isinstance(content, str):
                            # 使用空格分割计算单词数
                            token_count = len(content.split())
                            if token_count > 0:  # 过滤掉 0 token
                                lengths.append(token_count)
                                found_count += 1
                            else:
                                zero_count += 1
                            break  # 只取第一个 assistant 回复
        else:
            missing_count += 1
    
    print(f"提取完成，共提取 {len(lengths)} 个 decode length")
    print(f"  统计: 有效数据 {found_count}, 0 token 数据 {zero_count}, 缺失数据 {missing_count}")
    return lengths


def extract_decode_lengths_sharegpt(data: List[Dict[str, Any]]) -> List[int]:
    """从 shareGPT 数据中提取 decode length"""
    lengths = []
    total_items = len(data)
    print(f"正在提取 decode length，共 {total_items} 条记录...")
    
    for idx, item in enumerate(data):
        if (idx + 1) % 10000 == 0:
            print(f"  处理进度: {idx + 1}/{total_items} ({100*(idx+1)/total_items:.1f}%)")
        
        if 'conversations' in item:
            # shareGPT 格式通常是 conversations 列表
            for conv in item['conversations']:
                if conv.get('from') == 'gpt' or conv.get('role') == 'assistant':
                    text = conv.get('value', '') or conv.get('content', '')
                    if text and isinstance(text, str):
                        # 使用空格分割计算单词数
                        token_count = len(text.split())
                        if token_count > 0:  # 只记录非空回复
                            lengths.append(token_count)
    
    print(f"提取完成，共提取 {len(lengths)} 个 decode length")
    return lengths


def extract_decode_lengths_burstgpt(ds) -> List[int]:
    """直接从 burstGPT 数据集统计 response_tokens"""
    total_items = len(ds)
    print(f"正在统计 burstGPT response_tokens，共 {total_items} 条记录...")
    
    # 调试：查看第一条数据的字段
    if total_items > 0:
        first_item = ds[0]
        if isinstance(first_item, dict):
            print(f"  第一条数据的字段: {list(first_item.keys())[:10]}")  # 只显示前10个字段
        else:
            print(f"  第一条数据类型: {type(first_item)}")
    
    # 确定 response_tokens 字段名
    field_name = None
    if total_items > 0:
        first_item = ds[0]
        if isinstance(first_item, dict):
            # 尝试不同的字段名变体
            possible_keys = ['Response tokens', 'response_tokens', 'response tokens', 
                            'Response_tokens', 'ResponseTokens', 'responseTokens',
                            'Response Tokens', 'RESPONSE_TOKENS']
            for key in possible_keys:
                if key in first_item:
                    field_name = key
                    print(f"  找到字段名: {field_name}")
                    break
            
            # 如果还没找到，尝试模糊匹配
            if field_name is None:
                for key in first_item.keys():
                    key_lower = key.lower().replace('_', ' ').replace('-', ' ')
                    if 'response' in key_lower and 'token' in key_lower:
                        field_name = key
                        print(f"  找到字段名（模糊匹配）: {field_name}")
                        break
    
    if field_name is None:
        raise ValueError("无法找到 response_tokens 字段，请检查数据集格式")
    
    # 直接从数据集统计，不转换为列表
    lengths = []
    found_count = 0
    zero_count = 0
    missing_count = 0
    
    for idx in range(total_items):
        if (idx + 1) % 10000 == 0:
            print(f"  处理进度: {idx + 1}/{total_items} ({100*(idx+1)/total_items:.1f}%) - 已找到: {found_count}, 0 token: {zero_count}, 缺失: {missing_count}")
        
        item = ds[idx]
        token_count = item.get(field_name) if isinstance(item, dict) else None
        
        # 处理 token_count
        if token_count is not None:
            if isinstance(token_count, (int, float, np.integer, np.floating)):
                token_count = int(token_count)
                if token_count > 0:  # 过滤掉 0 token
                    lengths.append(token_count)
                    found_count += 1
                else:
                    zero_count += 1
            elif isinstance(token_count, str):
                try:
                    token_count = int(token_count)
                    if token_count > 0:  # 过滤掉 0 token
                        lengths.append(token_count)
                        found_count += 1
                    else:
                        zero_count += 1
                except (ValueError, TypeError):
                    missing_count += 1
        else:
            missing_count += 1
    
    print(f"统计完成，共提取 {len(lengths)} 个 decode length")
    print(f"  统计: 有效数据 {found_count}, 0 token 数据 {zero_count}, 缺失数据 {missing_count}")
    return lengths


def analyze_distribution(lengths: List[int], dataset_name: str) -> Dict[str, float]:
    """分析长度分布"""
    if not lengths:
        return {}
    
    lengths_array = np.array(lengths)
    
    # 分离 0-1000 和 >1000 的数据
    lengths_0_1000 = lengths_array[lengths_array <= 1000]
    lengths_over_1000 = lengths_array[lengths_array > 1000]
    
    stats = {
        'mean': float(np.mean(lengths_0_1000)) if len(lengths_0_1000) > 0 else 0.0,
        'median': float(np.median(lengths_0_1000)) if len(lengths_0_1000) > 0 else 0.0,
        'std': float(np.std(lengths_0_1000)) if len(lengths_0_1000) > 0 else 0.0,
        'min': int(np.min(lengths_0_1000)) if len(lengths_0_1000) > 0 else 0,
        'max': int(np.max(lengths_0_1000)) if len(lengths_0_1000) > 0 else 0,
        'p25': float(np.percentile(lengths_0_1000, 25)) if len(lengths_0_1000) > 0 else 0.0,
        'p75': float(np.percentile(lengths_0_1000, 75)) if len(lengths_0_1000) > 0 else 0.0,
        'p90': float(np.percentile(lengths_0_1000, 90)) if len(lengths_0_1000) > 0 else 0.0,
        'p95': float(np.percentile(lengths_0_1000, 95)) if len(lengths_0_1000) > 0 else 0.0,
        'p99': float(np.percentile(lengths_0_1000, 99)) if len(lengths_0_1000) > 0 else 0.0,
        'total_samples': len(lengths),
        'samples_0_1000': len(lengths_0_1000),
        'samples_over_1000': len(lengths_over_1000),
        'max_over_1000': int(np.max(lengths_over_1000)) if len(lengths_over_1000) > 0 else 0
    }
    
    print(f"\n{dataset_name} 统计信息:")
    print(f"  样本总数: {stats['total_samples']}")
    print(f"  0-1000 范围样本数: {stats['samples_0_1000']}")
    print(f"  >1000 样本数: {stats['samples_over_1000']}")
    if stats['samples_over_1000'] > 0:
        print(f"  >1000 最大长度: {stats['max_over_1000']} tokens")
    print(f"  平均长度 (0-1000): {stats['mean']:.2f} tokens")
    print(f"  中位数 (0-1000): {stats['median']:.2f} tokens")
    print(f"  标准差 (0-1000): {stats['std']:.2f} tokens")
    print(f"  最小值: {stats['min']} tokens")
    print(f"  最大值 (0-1000): {stats['max']} tokens")
    print(f"  P25: {stats['p25']:.2f} tokens")
    print(f"  P75: {stats['p75']:.2f} tokens")
    print(f"  P90: {stats['p90']:.2f} tokens")
    print(f"  P95: {stats['p95']:.2f} tokens")
    print(f"  P99: {stats['p99']:.2f} tokens")
    
    return stats


def plot_single_distribution(lengths: List[int], dataset_name: str, output_dir: str = './') -> None:
    """为单个数据集绘制分布图"""
    # 过滤数据：只保留 0-1000 范围用于绘图
    filtered = [l for l in lengths if l <= 1000]
    over_1000 = sum(1 for l in lengths if l > 1000)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'Decode Length Distribution: {dataset_name}', fontsize=16)
    
    # 1. 直方图 (0-1000，>1000归为一类)
    ax1 = axes[0]
    
    # 创建bins：0-1000，每20一个bin
    bins = list(range(0, 1001, 20))
    
    if filtered:
        counts, bin_edges = np.histogram(filtered, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        color = 'blue' if 'shareGPT' in dataset_name else 'red'
        ax1.bar(bin_centers, counts, width=18, alpha=0.7, 
                label=f'{dataset_name} (0-1000: {len(filtered)})', 
                color=color, align='center')
    
    # 在1000位置添加>1000的bar
    if over_1000 > 0:
        color = 'blue' if 'shareGPT' in dataset_name else 'red'
        ax1.bar(1000, over_1000, width=20, alpha=0.7, color=color, align='center', 
                label=f'{dataset_name} (>1000: {over_1000})')
        ax1.text(1000, over_1000 * 0.5, '>1000', 
                ha='center', va='bottom', fontsize=9, rotation=90)
    
    ax1.set_xlabel('Decode Length (tokens)')
    ax1.set_ylabel('Number')
    ax1.set_title('Length Distribution Histogram (0-1000, >1000 grouped)')
    ax1.set_xlim(-10, 1020)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图 (0-1000)
    ax2 = axes[1]
    if filtered:
        color = 'lightblue' if 'shareGPT' in dataset_name else 'lightcoral'
        bp = ax2.boxplot([filtered], labels=[f'{dataset_name}\n(>1000: {over_1000})'], 
                        patch_artist=True)
        bp['boxes'][0].set_facecolor(color)
        ax2.set_ylabel('Decode Length (tokens)')
        ax2.set_title('Box Plot (0-1000)')
        ax2.set_ylim(0, 1000)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = dataset_name.replace(' ', '_').lower()
    output_path = os.path.join(output_dir, f'{safe_name}_decode_length_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{dataset_name} 图表已保存至: {output_path}")
    print(f"{dataset_name} >1000 样本数: {over_1000}")
    plt.close()


def plot_distribution(sharegpt_lengths: List[int], burstgpt_lengths: List[int], 
                     lmsys_chat_lengths: Optional[List[int]] = None,
                     wildchat_lengths: Optional[List[int]] = None,
                     output_dir: str = './') -> None:
    """绘制分布图（数据集一左一右，只显示柱状图）"""
    datasets = []
    if sharegpt_lengths:
        datasets.append(('shareGPT', sharegpt_lengths, 'blue'))
    if burstgpt_lengths:
        datasets.append(('burstGPT', burstgpt_lengths, 'red'))
    if lmsys_chat_lengths:
        datasets.append(('lmsys-chat-1m', lmsys_chat_lengths, 'green'))
    if wildchat_lengths:
        datasets.append(('WildChat-1M', wildchat_lengths, 'orange'))
    
    if not datasets:
        print("没有数据可绘制")
        return
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle('Decode Length Distribution', fontsize=16)
    
    # 创建bins：0-1000，每20一个bin
    bins = list(range(0, 1001, 20))
    
    # 为每个数据集绘制柱状图
    for idx, (name, lengths, color) in enumerate(datasets):
        filtered = [l for l in lengths if l <= 1000]
        over_1000 = sum(1 for l in lengths if l > 1000)
        
        ax = axes[idx]
        if filtered:
            counts, bin_edges = np.histogram(filtered, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            ax.bar(bin_centers, counts, width=18, alpha=0.7, 
                    label=f'{name} (0-1000: {len(filtered)})', 
                    color=color, align='center')
        
        # 在1000位置添加>1000的bar
        if over_1000 > 0:
            ax.bar(1000, over_1000, width=20, alpha=0.7, color=color, align='center', 
                    label=f'{name} (>1000: {over_1000})')
            ax.text(1000, over_1000 * 0.5, '>1000', 
                    ha='center', va='bottom', fontsize=9, rotation=90)
        
        ax.set_xlabel('Decode Length (tokens)')
        ax.set_ylabel('Number')
        ax.set_title(name)
        ax.set_xlim(-10, 1020)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'decode_length_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    for name, lengths, _ in datasets:
        over_1000 = sum(1 for l in lengths if l > 1000)
        print(f"{name} >1000 样本数: {over_1000}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='下载并分析 shareGPT 和 burstGPT 的 decode length 分布')
    parser.add_argument('--sharegpt-url', type=str, help='shareGPT 数据下载 URL')
    parser.add_argument('--burstgpt-url', type=str, help='burstGPT 数据下载 URL')
    parser.add_argument('--sharegpt-file', type=str, help='shareGPT 数据文件或目录路径（如果已下载）')
    parser.add_argument('--burstgpt-file', type=str, help='burstGPT 数据文件或目录路径（如果已下载）')
    parser.add_argument('--sharegpt-dir', type=str, help='shareGPT 数据集目录路径（HuggingFace 格式）')
    parser.add_argument('--burstgpt-hf', type=str, help='从 HuggingFace 加载 burstGPT 数据集（数据集名称，如 lzzmm/BurstGPT）')
    parser.add_argument('--burstgpt-hf-split', type=str, default=None, help='HuggingFace 数据集的 split（如 train, test 等）')
    parser.add_argument('--lmsys-chat-hf', type=str, help='从 HuggingFace 加载 lmsys-chat-1m 数据集（数据集名称，如 lmsys/lmsys-chat-1m）')
    parser.add_argument('--lmsys-chat-hf-split', type=str, default=None, help='lmsys-chat-1m 数据集的 split（如 train, test 等）')
    parser.add_argument('--wildchat-hf', type=str, help='从 HuggingFace 加载 WildChat-1M 数据集（数据集名称，如 allenai/WildChat-1M）')
    parser.add_argument('--wildchat-hf-split', type=str, default=None, help='WildChat-1M 数据集的 split（如 train, test 等）')
    parser.add_argument('--output-dir', type=str, default='./', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    sharegpt_lengths = []
    burstgpt_lengths = []
    lmsys_chat_lengths = []
    wildchat_lengths = []
    
    # 处理 shareGPT 数据
    if args.sharegpt_dir:
        sharegpt_data = load_sharegpt_data(args.sharegpt_dir)
        sharegpt_lengths = extract_decode_lengths_sharegpt(sharegpt_data)
    elif args.sharegpt_file:
        sharegpt_data = load_sharegpt_data(args.sharegpt_file)
        sharegpt_lengths = extract_decode_lengths_sharegpt(sharegpt_data)
    elif args.sharegpt_url:
        sharegpt_path = os.path.join(args.output_dir, 'sharegpt_data.json')
        if not os.path.exists(sharegpt_path):
            download_file(args.sharegpt_url, sharegpt_path)
        sharegpt_data = load_sharegpt_data(sharegpt_path)
        sharegpt_lengths = extract_decode_lengths_sharegpt(sharegpt_data)
    
    # 处理 burstGPT 数据
    if args.burstgpt_hf:
        # 从 HuggingFace 加载，直接使用数据集对象
        burstgpt_ds = load_burstgpt_from_hf(args.burstgpt_hf, args.burstgpt_hf_split)
        burstgpt_lengths = extract_decode_lengths_burstgpt(burstgpt_ds)
    elif args.burstgpt_file:
        # 从文件加载，需要转换为类似数据集的格式
        burstgpt_data = load_burstgpt_data(args.burstgpt_file)
        # 创建一个简单的包装类来模拟数据集接口
        class SimpleDataset:
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        burstgpt_ds = SimpleDataset(burstgpt_data)
        burstgpt_lengths = extract_decode_lengths_burstgpt(burstgpt_ds)
    elif args.burstgpt_url:
        burstgpt_path = os.path.join(args.output_dir, 'burstgpt_data.json')
        if not os.path.exists(burstgpt_path):
            download_file(args.burstgpt_url, burstgpt_path)
        burstgpt_data = load_burstgpt_data(burstgpt_path)
        burstgpt_lengths = extract_decode_lengths_burstgpt(burstgpt_data)
    
    # 处理 lmsys-chat-1m 数据
    if args.lmsys_chat_hf:
        lmsys_chat_ds = load_lmsys_chat_from_hf(args.lmsys_chat_hf, args.lmsys_chat_hf_split)
        lmsys_chat_lengths = extract_decode_lengths_lmsys_chat(lmsys_chat_ds)
    
    # 处理 WildChat-1M 数据
    if args.wildchat_hf:
        wildchat_ds = load_wildchat_from_hf(args.wildchat_hf, args.wildchat_hf_split)
        wildchat_lengths = extract_decode_lengths_wildchat(wildchat_ds)
    
    # 分析分布
    if sharegpt_lengths:
        sharegpt_stats = analyze_distribution(sharegpt_lengths, 'shareGPT')
        stats_path = os.path.join(args.output_dir, 'sharegpt_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(sharegpt_stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存至: {stats_path}")
    
    if burstgpt_lengths:
        burstgpt_stats = analyze_distribution(burstgpt_lengths, 'burstGPT')
        stats_path = os.path.join(args.output_dir, 'burstgpt_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(burstgpt_stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存至: {stats_path}")
    
    if lmsys_chat_lengths:
        lmsys_chat_stats = analyze_distribution(lmsys_chat_lengths, 'lmsys-chat-1m')
        stats_path = os.path.join(args.output_dir, 'lmsys_chat_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(lmsys_chat_stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存至: {stats_path}")
    
    if wildchat_lengths:
        wildchat_stats = analyze_distribution(wildchat_lengths, 'WildChat-1M')
        stats_path = os.path.join(args.output_dir, 'wildchat_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(wildchat_stats, f, indent=2, ensure_ascii=False)
        print(f"统计信息已保存至: {stats_path}")
    
    # 绘制分布图
    if sharegpt_lengths or burstgpt_lengths or lmsys_chat_lengths or wildchat_lengths:
        plot_distribution(sharegpt_lengths, burstgpt_lengths, lmsys_chat_lengths, wildchat_lengths, args.output_dir)
    
    # 对比分析
    if sharegpt_lengths and burstgpt_lengths:
        print("\n=== 对比分析 ===")
        sharegpt_mean = np.mean(sharegpt_lengths)
        burstgpt_mean = np.mean(burstgpt_lengths)
        print(f"shareGPT 平均长度: {sharegpt_mean:.2f} tokens")
        print(f"burstGPT 平均长度: {burstgpt_mean:.2f} tokens")
        print(f"差异: {abs(sharegpt_mean - burstgpt_mean):.2f} tokens")
        print(f"比例: {sharegpt_mean / burstgpt_mean:.2f}x")


if __name__ == '__main__':
    main()

