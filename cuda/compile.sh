#!/bin/bash
# CUDA 编译脚本

# 设置 nvcc 路径（可选，因为 /usr/bin 已在 PATH 中）
NVCC="/usr/bin/nvcc"

# 检查 nvcc 是否存在
if [ ! -f "$NVCC" ]; then
    echo "错误: 找不到 nvcc 在 $NVCC"
    exit 1
fi

# 获取文件名（不带扩展名）
if [ -z "$1" ]; then
    echo "用法: $0 <源文件.cu>"
    exit 1
fi

SOURCE_FILE="$1"
OUTPUT_FILE="${SOURCE_FILE%.cu}"

# 编译
echo "正在编译 $SOURCE_FILE ..."
$NVCC "$SOURCE_FILE" -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "编译成功！输出文件: $OUTPUT_FILE"
else
    echo "编译失败！"
    exit 1
fi


