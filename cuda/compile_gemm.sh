#!/bin/bash
# 尝试多种方式找到 nvcc
NVCC=""
for path in /usr/local/cuda*/bin/nvcc /opt/cuda*/bin/nvcc /usr/bin/nvcc $(which nvcc 2>/dev/null); do
    if [ -x "$path" ]; then
        NVCC="$path"
        break
    fi
done

if [ -z "$NVCC" ]; then
    echo "错误: 未找到 nvcc 编译器"
    echo "请确保 CUDA 已正确安装，或者设置 CUDA_HOME 环境变量"
    exit 1
fi

echo "使用 nvcc: $NVCC"
$NVCC -o GEMM GEMM.cu -O3
if [ $? -eq 0 ]; then
    echo "编译成功！"
    chmod +x GEMM
else
    echo "编译失败"
    exit 1
fi
