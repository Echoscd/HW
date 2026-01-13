#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

// 定义线程块大小
#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

// 索引宏：计算行优先存储的索引
#define INDX(row, col, ld) ((row) * (ld) + (col))

// 使用共享内存的矩阵转置kernel
__global__ void matrix_transpose_shared(float* a, float* c, int m, int n)
{
    // 声明静态分配的共享内存数组
    __shared__ float smemArray[THREADS_PER_BLOCK_X][THREADS_PER_BLOCK_Y+1];
    
    // 确定当前线程块对应的tile的行和列索引
    const int tileCol = blockDim.x * blockIdx.x;
    const int tileRow = blockDim.y * blockIdx.y;
    
    // 计算全局索引
    int globalRow = tileRow + threadIdx.y;
    int globalCol = tileCol + threadIdx.x;
    
    // 从全局内存读取到共享内存数组
    // 注意：读取时使用 (tileRow + threadIdx.y, tileCol + threadIdx.x)
    // 写入共享内存时使用 [threadIdx.y][threadIdx.x]
    if (globalRow < m && globalCol < n) {
        smemArray[threadIdx.y][threadIdx.x] = a[INDX(globalRow, globalCol, n)];
    }
    
    // 同步线程块中的所有线程
    __syncthreads();
    
    // 从共享内存写入结果到全局内存（转置）
    // 注意：读取共享内存时使用 [threadIdx.x][threadIdx.y]（交换索引）
    // 写入全局内存时使用 (tileCol + threadIdx.y, tileRow + threadIdx.x)（交换行列）
    int transRow = tileCol + threadIdx.y;
    int transCol = tileRow + threadIdx.x;
    
    if (transRow < n && transCol < m) {
        c[INDX(transRow, transCol, m)] = smemArray[threadIdx.x][threadIdx.y];
    }
}

int main()
{
    // 定义矩阵维度
    const int ROWS = 32768;
    const int COLS = 32768;
    const int SIZE = ROWS * COLS;
    const size_t BYTES = SIZE * sizeof(float);
    
    cout << "========== 矩阵转置程序 ==========" << endl;
    cout << "矩阵维度: " << ROWS << " x " << COLS << endl;
    cout << "转置后维度: " << COLS << " x " << ROWS << endl;
    cout << endl;
    
    // 1. 在CPU上定义并初始化矩阵
    float *h_input, *h_output;
    h_input = (float*)malloc(BYTES);
    h_output = (float*)malloc(COLS * ROWS * sizeof(float));
    
    // 初始化输入矩阵
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            h_input[i * COLS + j] = i * COLS + j + 1;  // 填充1到SIZE的值
        }
    }
    
    // 只显示矩阵的前5x5部分
    cout << "========== 输入矩阵（前5x5） ==========" << endl;
    int displayRows = (ROWS < 5) ? ROWS : 5;
    int displayCols = (COLS < 5) ? COLS : 5;
    for (int i = 0; i < displayRows; i++) {
        for (int j = 0; j < displayCols; j++) {
            cout << h_input[i * COLS + j] << "\t";
        }
        cout << endl;
    }
    cout << "... (矩阵大小: " << ROWS << " x " << COLS << ")" << endl;
    cout << endl;
    
    // 2. 在GPU上分配内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, BYTES);
    cudaMalloc(&d_output, COLS * ROWS * sizeof(float));
    
    // 3. 将矩阵从CPU拷贝到GPU（统计传输时间）
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float elapsedTime;
    
    // 统计Host to Device传输时间
    cudaEventRecord(start, 0);
    cudaMemcpy(d_input, h_input, BYTES, cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "========== 时间统计 ==========" << endl;
    cout << "Host to Device 传输时间: " << elapsedTime << " ms" << endl;
    
    // 4. 配置kernel执行参数
    // 使用定义的线程块大小
    dim3 blockSize(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, 
                  (ROWS + blockSize.y - 1) / blockSize.y);
    
    cout << "========== Kernel配置 ==========" << endl;
    cout << "线程块大小: (" << blockSize.x << ", " << blockSize.y << ")" << endl;
    cout << "网格大小: (" << gridSize.x << ", " << gridSize.y << ")" << endl;
    cout << "共享内存大小: " << (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y * sizeof(float)) 
         << " 字节 (" << (THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_Y * sizeof(float) / 1024.0) << " KB)" << endl;
    cout << endl;
    
    // 5. 在GPU上执行转置kernel（使用共享内存版本，统计kernel执行时间）
    cudaEventRecord(start, 0);
    // 注意：参数顺序为 (input, output, rows, cols)
    // 在kernel中，m是行数，n是列数
    matrix_transpose_shared<<<gridSize, blockSize>>>(d_input, d_output, ROWS, COLS);
    
    // 检查kernel执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "Kernel启动错误: " << cudaGetErrorString(err) << endl;
        return -1;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU Kernel 执行时间: " << elapsedTime << " ms" << endl;
    
    // 等待kernel执行完成
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cout << "Kernel执行错误: " << cudaGetErrorString(err) << endl;
        return -1;
    }
    
    // 6. 将结果从GPU拷贝回CPU（统计传输时间）
    cudaEventRecord(start, 0);
    cudaMemcpy(h_output, d_output, COLS * ROWS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Device to Host 传输时间: " << elapsedTime << " ms" << endl;
    cout << endl;
    
    // 7. 输出转置后的矩阵（只显示前5x5部分）
    cout << "========== 转置后的矩阵（前5x5） ==========" << endl;
    int displayTransRows = (COLS < 5) ? COLS : 5;
    int displayTransCols = (ROWS < 5) ? ROWS : 5;
    for (int i = 0; i < displayTransRows; i++) {
        for (int j = 0; j < displayTransCols; j++) {
            cout << h_output[i * ROWS + j] << "\t";
        }
        cout << endl;
    }
    cout << "... (矩阵大小: " << COLS << " x " << ROWS << ")" << endl;
    cout << endl;
    
    // 8. 释放内存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cout << "========== 程序执行完成 ==========" << endl;
    
    return 0;
}
