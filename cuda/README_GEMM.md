# GEMM.cu 编译和运行指南

## 编译方法

### 方法 1: 使用 nvcc（如果已安装）
```bash
nvcc -o GEMM GEMM.cu -O3
```

### 方法 2: 如果 nvcc 不在 PATH 中
```bash
# 设置 CUDA 路径（根据实际安装位置调整）
export CUDA_HOME=/usr/local/cuda  # 或其他 CUDA 安装路径
export PATH=$CUDA_HOME/bin:$PATH

# 然后编译
nvcc -o GEMM GEMM.cu -O3
```

### 方法 3: 使用完整路径
```bash
/usr/local/cuda/bin/nvcc -o GEMM GEMM.cu -O3
# 或
/opt/cuda/bin/nvcc -o GEMM GEMM.cu -O3
```

## 运行方法

编译成功后，直接运行：
```bash
./GEMM
```

## 程序说明

这个程序实现了优化的矩阵乘法（GEMM）kernel，会：
1. 首先测试正确性（M=N=K=512）
2. 然后测试不同矩阵大小的性能（从 128x128 到 16384x16384）
3. 输出每个测试的性能（Gflops）

## 注意事项

- 确保系统已安装 CUDA Toolkit
- 确保有可用的 NVIDIA GPU
- 如果编译失败，请检查 CUDA 版本是否兼容
