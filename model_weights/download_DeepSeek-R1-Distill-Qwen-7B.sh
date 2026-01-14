#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# 下载 DeepSeek-R1-Distill-Qwen-7B 模型权重
#

set -e

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_SCRIPT="${SCRIPT_DIR}/download_model.py"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR="${SCRIPT_DIR}/DeepSeek-R1-Distill-Qwen-7B"

echo "=========================================="
echo "下载模型: ${MODEL_NAME}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# 检查 Python 脚本是否存在
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "错误: 找不到下载脚本 $DOWNLOAD_SCRIPT"
    exit 1
fi

# 执行下载
python3 "$DOWNLOAD_SCRIPT" \
    --model-name "$MODEL_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --revision main

echo ""
echo "=========================================="
echo "✓ 下载完成！"
echo "模型路径: ${OUTPUT_DIR}"
echo "=========================================="

