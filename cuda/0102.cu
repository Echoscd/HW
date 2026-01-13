#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;
__global__ void vectoradd(float* A, float* B, float* C, int length)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < length) {
        C[id] = A[id] + B[id];
    }
}
__global__ void example_syncthreads(int* input_data, int* output_data) {
    __shared__ int shared_data[128];
    // Every thread writes to a distinct element of 'shared_data':
    shared_data[threadIdx.x] = threadIdx.x + 1;;

    // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered 
    // before any thread is unblocked from '__syncthreads()':
    //__syncthreads();

    // A single thread safely reads 'shared_data':
    if (threadIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_data[i];
        }
        output_data[threadIdx.x] = sum;
    }
}
int main()
{
    // 查询 GPU 设备属性
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "系统中 CUDA 设备数量: " << deviceCount << endl << endl;
    
    if (deviceCount == 0) {
        cout << "错误: 未找到 CUDA 设备！" << endl;
        return -1;
    }
    
    // 获取当前设备属性
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    cout << "========== GPU 设备信息 ==========" << endl;
    cout << "设备名称: " << prop.name << endl;
    cout << "计算能力: " << prop.major << "." << prop.minor << endl;
    cout << "多处理器数量: " << prop.multiProcessorCount << endl;
    cout << endl;
    
    cout << "========== 共享内存信息 ==========" << endl;
    cout << "每个多处理器的共享内存: " << prop.sharedMemPerMultiprocessor 
         << " 字节 (" << prop.sharedMemPerMultiprocessor / 1024.0 << " KB)" << endl;
    cout << "每个线程块的共享内存: " << prop.sharedMemPerBlock 
         << " 字节 (" << prop.sharedMemPerBlock / 1024.0 << " KB)" << endl;
    cout << "每个线程块的最大共享内存: " << prop.sharedMemPerBlockOptin 
         << " 字节 (" << prop.sharedMemPerBlockOptin / 1024.0 << " KB)" << endl;
    cout << endl;
    
    cout << "========== 线程和块限制 ==========" << endl;
    cout << "每个线程块的最大线程数: " << prop.maxThreadsPerBlock << endl;
    cout << "每个线程块的最大维度: (" << prop.maxThreadsDim[0] << ", " 
         << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
    cout << "每个网格的最大维度: (" << prop.maxGridSize[0] << ", " 
         << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
    cout << "每个多处理器的最大线程数: " << prop.maxThreadsPerMultiProcessor << endl;
    cout << endl;
    
    cout << "========== 内存信息 ==========" << endl;
    cout << "全局内存总量: " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << endl;
    cout << "常量内存: " << prop.totalConstMem / 1024.0 << " KB" << endl;
    cout << "内存总线宽度: " << prop.memoryBusWidth << " 位" << endl;
    cout << "内存时钟频率: " << prop.memoryClockRate / 1000.0 << " MHz" << endl;
    cout << endl;
    
    cout << "========== 其他信息 ==========" << endl;
    cout << "时钟频率: " << prop.clockRate / 1000.0 << " MHz" << endl;
    cout << "L2 缓存大小: " << prop.l2CacheSize / 1024.0 << " KB" << endl;
    cout << "纹理对齐: " << prop.textureAlignment << " 字节" << endl;
    cout << "warp 大小: " << prop.warpSize << endl;
    cout << "=================================" << endl << endl;
    
    // 检查当前使用的共享内存
    size_t sharedMemUsed = 128 * sizeof(int);  // 当前 kernel 使用的共享内存
    cout << "当前 kernel 使用的共享内存: " << sharedMemUsed << " 字节 (" 
         << sharedMemUsed / 1024.0 << " KB)" << endl;
    if (sharedMemUsed > prop.sharedMemPerBlock) {
        cout << "警告: 使用的共享内存超过每个线程块的限制！" << endl;
    } else {
        cout << "共享内存使用情况: " << (sharedMemUsed * 100.0 / prop.sharedMemPerBlock) 
             << "%" << endl;
    }
    cout << endl;
    
    dim3 blockSize(100,1);
    dim3 gridSize(1,1);
    int size = 100;
    size_t type = size * sizeof(int); 
    int *input, *output;
    cudaMallocHost(&input, type);
    cudaMallocHost(&output, type);
    int *devinput, *devoutput;
    cudaMalloc(&devinput, type);
    cudaMalloc(&devoutput, type);
    for (int i=0; i<size; i++){
        input[i] = i;
    }
    cudaMemcpy(devinput, input, type, cudaMemcpyHostToDevice);
    
    cout << "========== 执行 Kernel ==========" << endl;
    cout << "线程块配置: (" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << ")" << endl;
    cout << "每个线程块的线程数: (" << blockSize.x << ", " << blockSize.y << ", " << blockSize.z << ")" << endl;
    cout << "总线程数: " << (gridSize.x * gridSize.y * gridSize.z * blockSize.x * blockSize.y * blockSize.z) << endl;
    cout << endl;
    
    example_syncthreads<<<gridSize, blockSize>>>(devinput, devoutput);
    
    // 检查 kernel 执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "Kernel 启动错误: " << cudaGetErrorString(err) << endl;
    }
    
    cudaDeviceSynchronize();
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cout << "Kernel 执行错误: " << cudaGetErrorString(err) << endl;
    }
    
    cudaMemcpy(output, devoutput, type, cudaMemcpyDeviceToHost);
    
    cout << "========== 输出结果 ==========" << endl;
    for (int i=0; i<size; i++){
        cout << "output[" << i << "] = " << output[i] << endl;
    }
    
    // 释放内存
    cudaFreeHost(input);
    cudaFreeHost(output);
    cudaFree(devinput);
    cudaFree(devoutput);
    
    return 0;
}