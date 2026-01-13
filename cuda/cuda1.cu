#include <stdio.h>
#include <cuda_runtime.h>

// Kernel definition - 使用二维线程处理二维矩阵
__global__ void MatAdd(float* A, float* B, float* C, int width, int height)
{
    // 计算当前线程对应的矩阵位置（行和列）
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // 行索引
    
    // 边界检查：确保索引在有效范围内
    if (row < height && col < width) {
        // 将二维索引转换为线性索引（矩阵按行存储）
        int idx = row * width + col;
        
        // 执行矩阵加法
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    // 矩阵维度
    int M = 16;  // 行数
    int N = 16;  // 列数
    size_t size = M * N * sizeof(float);
    
    // 分配主机内存
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // 初始化矩阵数据
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            h_A[idx] = 1.0f;
            h_B[idx] = 2.0f;
        }
    }
    
    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    
    // 将数据复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 配置二维线程块和网格
    dim3 blockSize(8, 8);  // 每个线程块是 8x8 = 64 个线程
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // Kernel invocation - 使用二维线程配置处理二维矩阵
    MatAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M);
    
    // 等待 GPU 完成计算
    cudaDeviceSynchronize();
    
    // 将结果复制回主机
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    // 打印矩阵结果（显示整个矩阵）
    printf("矩阵加法结果 (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            printf("%6.1f ", h_C[idx]);
        }
        printf("\n");
    }
    
    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}