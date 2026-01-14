#!/bin/bash
# GEMM.cu 编译和运行脚本

echo "========== 查找 nvcc 编译器 =========="

# 尝试多种方式找到 nvcc
NVCC=""
POSSIBLE_PATHS=(
    "/usr/local/cuda/bin/nvcc"
    "/opt/cuda/bin/nvcc"
    "/usr/cuda/bin/nvcc"
    "$(which nvcc 2>/dev/null)"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    if [ -n "$path" ] && [ -x "$path" ]; then
        NVCC="$path"
        echo "找到 nvcc: $NVCC"
        $NVCC --version | head -1
        break
    fi
done

if [ -z "$NVCC" ]; then
    echo ""
    echo "错误: 未找到 nvcc 编译器"
    echo ""
    echo "请尝试以下方法之一："
    echo "1. 设置 CUDA 环境变量："
    echo "   export CUDA_HOME=/usr/local/cuda"
    echo "   export PATH=\$CUDA_HOME/bin:\$PATH"
    echo ""
    echo "2. 使用完整路径编译："
    echo "   /usr/local/cuda/bin/nvcc -o GEMM GEMM.cu -O3"
    echo ""
    echo "3. 如果使用环境模块系统："
    echo "   module load cuda"
    echo "   nvcc -o GEMM GEMM.cu -O3"
    echo ""
    exit 1
fi

echo ""
echo "========== 编译 GEMM.cu =========="
$NVCC -o GEMM GEMM.cu -O3

if [ $? -eq 0 ]; then
    echo "编译成功！"
    chmod +x GEMM
    echo ""
    echo "========== 运行 GEMM =========="
    ./GEMM
else
    echo "编译失败，请检查错误信息"
    exit 1
fi





