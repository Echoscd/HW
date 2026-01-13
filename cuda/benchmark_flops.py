"""
GPU Peak FLOPs Benchmark using PyTorch
测量 GPU 的极限浮点计算能力
"""

import torch
import torch.utils.benchmark as benchmark
from typing import Tuple
import argparse


def measure_matmul_flops(
    M: int, N: int, K: int,
    dtype: torch.dtype,
    device: str = "cuda",
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> Tuple[float, float, float]:
    """
    测量矩阵乘法的 FLOPs
    
    Returns:
        (achieved_tflops, time_ms, arithmetic_intensity)
    """
    # 创建矩阵
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)
    
    # 计算理论值
    flops = 2 * M * N * K
    bytes_accessed = (M * K + K * N + M * N) * A.element_size()
    arithmetic_intensity = flops / bytes_accessed
    
    # 预热
    for _ in range(warmup_iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(benchmark_iters):
        C = torch.matmul(A, B)
    
    end_event.record()
    torch.cuda.synchronize()
    
    # 计算结果
    elapsed_ms = start_event.elapsed_time(end_event) / benchmark_iters
    achieved_tflops = (flops / 1e12) / (elapsed_ms / 1000)
    
    return achieved_tflops, elapsed_ms, arithmetic_intensity


def measure_fma_flops(
    size: int,
    dtype: torch.dtype,
    device: str = "cuda",
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
) -> Tuple[float, float]:
    """
    使用 FMA (Fused Multiply-Add) 测量纯计算 FLOPs
    C = A * B + C (每个元素 2 FLOPs)
    """
    A = torch.randn(size, size, dtype=dtype, device=device)
    B = torch.randn(size, size, dtype=dtype, device=device)
    C = torch.randn(size, size, dtype=dtype, device=device)
    
    flops_per_iter = 2 * size * size  # 每次 FMA 是 2 FLOPs
    
    # 预热
    for _ in range(warmup_iters):
        C = torch.addcmul(C, A, B)
    torch.cuda.synchronize()
    
    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    n_ops = 1000  # 内循环次数
    
    torch.cuda.synchronize()
    start_event.record()
    
    for _ in range(benchmark_iters):
        for _ in range(n_ops):
            C = torch.addcmul(C, A, B)
    
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event) / benchmark_iters
    total_flops = flops_per_iter * n_ops
    achieved_tflops = (total_flops / 1e12) / (elapsed_ms / 1000)
    
    return achieved_tflops, elapsed_ms


def run_benchmark(args):
    """运行完整的 benchmark"""
    
    print("=" * 60)
    print("GPU Peak FLOPs Benchmark")
    print("=" * 60)
    
    # GPU 信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"Memory: {gpu_mem:.1f} GB")
    else:
        print("No CUDA device available!")
        return
    
    print("=" * 60)
    
    # 数据类型配置
    dtype_configs = [
        ("FP32", torch.float32),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
    ]
    
    # 矩阵大小配置 (M, N, K) - 选择 Compute Bound 的配置
    # 确保维度是 8 的倍数以利用 Tensor Core
    size_configs = [
        (4096, 4096, 4096),
        (8192, 8192, 8192),
        (16384, 16384, 4096),
        (16384, 16384, 8192),
    ]
    
    print("\n[1] Matrix Multiplication Benchmark (GEMM)")
    print("-" * 60)
    print(f"{'Dtype':<8} {'M':>6} {'N':>6} {'K':>6} {'I':>8} {'Time(ms)':>10} {'TFLOPs':>10}")
    print("-" * 60)
    
    best_tflops = {}
    
    for dtype_name, dtype in dtype_configs:
        best_tflops[dtype_name] = 0
        
        for M, N, K in size_configs:
            try:
                tflops, time_ms, intensity = measure_matmul_flops(
                    M, N, K, dtype,
                    warmup_iters=args.warmup,
                    benchmark_iters=args.iters,
                )
                best_tflops[dtype_name] = max(best_tflops[dtype_name], tflops)
                print(f"{dtype_name:<8} {M:>6} {N:>6} {K:>6} {intensity:>8.1f} {time_ms:>10.3f} {tflops:>10.2f}")
            except RuntimeError as e:
                print(f"{dtype_name:<8} {M:>6} {N:>6} {K:>6} {'OOM':>8} {'-':>10} {'-':>10}")
        
        # 清理显存
        torch.cuda.empty_cache()
    
    print("-" * 60)
    print("\n[2] Peak TFLOPs Summary")
    print("-" * 40)
    for dtype_name, tflops in best_tflops.items():
        print(f"{dtype_name}: {tflops:.2f} TFLOPs")
    
    # 与理论峰值对比（如果是 H200）
    print("\n[3] Theoretical Peak (H200 Reference)")
    print("-" * 40)
    theoretical = {
        "FP32": 67,      # TFLOPs
        "FP16": 990,     # TFLOPs (Tensor Core)
        "BF16": 990,     # TFLOPs (Tensor Core)
    }
    
    for dtype_name in best_tflops:
        if dtype_name in theoretical:
            efficiency = best_tflops[dtype_name] / theoretical[dtype_name] * 100
            print(f"{dtype_name}: {best_tflops[dtype_name]:.2f} / {theoretical[dtype_name]} TFLOPs ({efficiency:.1f}%)")
    
    print("=" * 60)


def run_scaling_test(args):
    """测试不同矩阵大小的性能扩展"""
    
    print("\n[4] Scaling Test (BF16)")
    print("-" * 70)
    print(f"{'Size':>8} {'FLOPs(G)':>12} {'Bytes(MB)':>12} {'I':>8} {'Time(ms)':>10} {'TFLOPs':>10}")
    print("-" * 70)
    
    dtype = torch.bfloat16
    sizes = [1024, 2048, 4096, 8192, 12288, 16384]
    
    for size in sizes:
        try:
            tflops, time_ms, intensity = measure_matmul_flops(
                size, size, size, dtype,
                warmup_iters=args.warmup,
                benchmark_iters=args.iters,
            )
            flops_g = 2 * size**3 / 1e9
            bytes_mb = 3 * size**2 * 2 / 1e6  # BF16 = 2 bytes
            print(f"{size:>8} {flops_g:>12.1f} {bytes_mb:>12.1f} {intensity:>8.1f} {time_ms:>10.3f} {tflops:>10.2f}")
        except RuntimeError:
            print(f"{size:>8} {'OOM':>12}")
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU FLOPs Benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--scaling", action="store_true", help="Run scaling test")
    args = parser.parse_args()
    
    run_benchmark(args)
    
    if args.scaling:
        run_scaling_test(args)