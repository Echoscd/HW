#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define HALF2(pointer) (reinterpret_cast<half2*>(&(pointer))[0])

float testError(
    void (*gpuSgemm) (half *, half *, half *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K);
float testPerformance(
    void (*gpuSgemm) (half *, half *, half *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(
    half *a, half *b, half *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0f;
            for (int k = 0; k < K; k++) {
                psum += __half2float(a[OFFSET(m, k, K)]) * __half2float(b[OFFSET(k, n, N)]);
            }
            c[OFFSET(m, n, N)] = __float2half(psum);
        }
    }
}


__global__ void sgemm_V3(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int M, const int N, const int K) {

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ half s_a[2][BK][BM];
    __shared__ half s_b[2][BK][BN];

    half2 r_load_a[2];
    half2 r_load_b[2];
    half r_comp_a[TM];
    half r_comp_b[TN];
    float r_c[TM][TN] = {0.0f};

    int load_a_smem_m = tid >> 1;
    int load_a_smem_k = (tid & 1) << 2;
    int load_b_smem_k = tid >> 5;
    int load_b_smem_n = (tid & 31) << 2;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        HALF2(r_load_a[0]) = HALF2(a[load_a_gmem_addr]);
        HALF2(r_load_a[1]) = HALF2(a[load_a_gmem_addr + 2]);
        HALF2(r_load_b[0]) = HALF2(b[load_b_gmem_addr]);
        HALF2(r_load_b[1]) = HALF2(b[load_b_gmem_addr + 2]);

        s_a[0][load_a_smem_k    ][load_a_smem_m] = __low2half(r_load_a[0]);
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = __high2half(r_load_a[0]);
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = __low2half(r_load_a[1]);
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = __high2half(r_load_a[1]);
        HALF2(s_b[0][load_b_smem_k][load_b_smem_n]) = HALF2(r_load_b[0]);
        HALF2(s_b[0][load_b_smem_k][load_b_smem_n + 2]) = HALF2(r_load_b[1]);
    }

    for (int bk = 1; bk < (K + BK - 1) / BK; bk++) {

        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, K);
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, N);
        HALF2(r_load_a[0]) = HALF2(a[load_a_gmem_addr]);
        HALF2(r_load_a[1]) = HALF2(a[load_a_gmem_addr + 2]);
        HALF2(r_load_b[0]) = HALF2(b[load_b_gmem_addr]);
        HALF2(r_load_b[1]) = HALF2(b[load_b_gmem_addr + 2]);

        #pragma unroll
        for (int tk = 0; tk < BK; tk++) {
            // 读取 4 个 half 值到 r_comp_a[0..3]
            half2 temp_a0 = HALF2(s_a[smem_sel][tk][ty * TM / 2         ]);
            half2 temp_a1 = HALF2(s_a[smem_sel][tk][ty * TM / 2 + 2      ]);
            r_comp_a[0] = __low2half(temp_a0);
            r_comp_a[1] = __high2half(temp_a0);
            r_comp_a[2] = __low2half(temp_a1);
            r_comp_a[3] = __high2half(temp_a1);
            
            // 读取 4 个 half 值到 r_comp_a[4..7]
            half2 temp_a2 = HALF2(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            half2 temp_a3 = HALF2(s_a[smem_sel][tk][ty * TM / 2 + BM / 2 + 2]);
            r_comp_a[4] = __low2half(temp_a2);
            r_comp_a[5] = __high2half(temp_a2);
            r_comp_a[6] = __low2half(temp_a3);
            r_comp_a[7] = __high2half(temp_a3);
            
            // 读取 4 个 half 值到 r_comp_b[0..3]
            half2 temp_b0 = HALF2(s_b[smem_sel][tk][tx * TN / 2         ]);
            half2 temp_b1 = HALF2(s_b[smem_sel][tk][tx * TN / 2 + 2      ]);
            r_comp_b[0] = __low2half(temp_b0);
            r_comp_b[1] = __high2half(temp_b0);
            r_comp_b[2] = __low2half(temp_b1);
            r_comp_b[3] = __high2half(temp_b1);
            
            // 读取 4 个 half 值到 r_comp_b[4..7]
            half2 temp_b2 = HALF2(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);
            half2 temp_b3 = HALF2(s_b[smem_sel][tk][tx * TN / 2 + BN / 2 + 2]);
            r_comp_b[4] = __low2half(temp_b2);
            r_comp_b[5] = __high2half(temp_b2);
            r_comp_b[6] = __low2half(temp_b3);
            r_comp_b[7] = __high2half(temp_b3);

            #pragma unroll
            for (int tm = 0; tm < TM; tm++) {
                #pragma unroll
                for (int tn = 0; tn < TN; tn++) {
                    r_c[tm][tn] += __half2float(r_comp_a[tm]) * __half2float(r_comp_b[tn]);
                }
            }
        }

        s_a[smem_sel_next][load_a_smem_k    ][load_a_smem_m] = __low2half(r_load_a[0]);
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = __high2half(r_load_a[0]);
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = __low2half(r_load_a[1]);
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = __high2half(r_load_a[1]);
        HALF2(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = HALF2(r_load_b[0]);
        HALF2(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n + 2]) = HALF2(r_load_b[1]);

        __syncthreads();
    }

    #pragma unroll
    for (int tk = 0; tk < BK; tk++) {
        // 读取 4 个 half 值到 r_comp_a[0..3]
        half2 temp_a0 = HALF2(s_a[1][tk][ty * TM / 2         ]);
        half2 temp_a1 = HALF2(s_a[1][tk][ty * TM / 2 + 2      ]);
        r_comp_a[0] = __low2half(temp_a0);
        r_comp_a[1] = __high2half(temp_a0);
        r_comp_a[2] = __low2half(temp_a1);
        r_comp_a[3] = __high2half(temp_a1);
        
        // 读取 4 个 half 值到 r_comp_a[4..7]
        half2 temp_a2 = HALF2(s_a[1][tk][ty * TM / 2 + BM / 2]);
        half2 temp_a3 = HALF2(s_a[1][tk][ty * TM / 2 + BM / 2 + 2]);
        r_comp_a[4] = __low2half(temp_a2);
        r_comp_a[5] = __high2half(temp_a2);
        r_comp_a[6] = __low2half(temp_a3);
        r_comp_a[7] = __high2half(temp_a3);
        
        // 读取 4 个 half 值到 r_comp_b[0..3]
        half2 temp_b0 = HALF2(s_b[1][tk][tx * TN / 2         ]);
        half2 temp_b1 = HALF2(s_b[1][tk][tx * TN / 2 + 2      ]);
        r_comp_b[0] = __low2half(temp_b0);
        r_comp_b[1] = __high2half(temp_b0);
        r_comp_b[2] = __low2half(temp_b1);
        r_comp_b[3] = __high2half(temp_b1);
        
        // 读取 4 个 half 值到 r_comp_b[4..7]
        half2 temp_b2 = HALF2(s_b[1][tk][tx * TN / 2 + BN / 2]);
        half2 temp_b3 = HALF2(s_b[1][tk][tx * TN / 2 + BN / 2 + 2]);
        r_comp_b[4] = __low2half(temp_b2);
        r_comp_b[5] = __high2half(temp_b2);
        r_comp_b[6] = __low2half(temp_b3);
        r_comp_b[7] = __high2half(temp_b3);

        #pragma unroll
        for (int tm = 0; tm < TM; tm++) {
            #pragma unroll
            for (int tn = 0; tn < TN; tn++) {
                r_c[tm][tn] += __half2float(r_comp_a[tm]) * __half2float(r_comp_b[tn]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        // 存储前 4 个值 (r_c[i][0..3])
        half2 h2_0 = __floats2half2_rn(r_c[i][0], r_c[i][1]);
        half2 h2_1 = __floats2half2_rn(r_c[i][2], r_c[i][3]);
        HALF2(c[store_c_gmem_addr]) = h2_0;
        HALF2(c[store_c_gmem_addr + 2]) = h2_1;
        // 存储后 4 个值 (r_c[i][4..7])
        half2 h2_4 = __floats2half2_rn(r_c[i][4], r_c[i][5]);
        half2 h2_5 = __floats2half2_rn(r_c[i][6], r_c[i][7]);
        HALF2(c[store_c_gmem_addr + BN / 2]) = h2_4;
        HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = h2_5;
    }
    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, N);
        // 存储前 4 个值 (r_c[i + TM/2][0..3])
        half2 h2_0 = __floats2half2_rn(r_c[i + TM / 2][0], r_c[i + TM / 2][1]);
        half2 h2_1 = __floats2half2_rn(r_c[i + TM / 2][2], r_c[i + TM / 2][3]);
        HALF2(c[store_c_gmem_addr]) = h2_0;
        HALF2(c[store_c_gmem_addr + 2]) = h2_1;
        // 存储后 4 个值 (r_c[i + TM/2][4..7])
        half2 h2_4 = __floats2half2_rn(r_c[i + TM / 2][4], r_c[i + TM / 2][5]);
        half2 h2_5 = __floats2half2_rn(r_c[i + TM / 2][6], r_c[i + TM / 2][7]);
        HALF2(c[store_c_gmem_addr + BN / 2]) = h2_4;
        HALF2(c[store_c_gmem_addr + BN / 2 + 2]) = h2_5;
    }
}



int main(void) {
    printf("\nKernal = sgemm_V3 (FP16)\n");
    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 128, BN = 128, TM = 8, TN = 8;
    void (*gpuSgemm) (half *, half *, half *, const int, const int, const int) = sgemm_V3;

    {
        const int M = 512, N = 512, K = 512;
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
        float max_error = testError(gpuSgemm, gridDim, blockDim, M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[1] = {16384};
    const int N_list[1] = {16384};
    const int K_list[1] = { 1024};
    
    const int TESTNUM = 1;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}


float testError(
    void (*gpuSgemm) (half *, half *, half *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (half *)malloc(size_a);
    h_b = (half *)malloc(size_b);
    h_c = (half *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (half *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = __float2half(rand() / float(RAND_MAX));
    for (int i = 0; i < K * N; i++)
        h_b[i] = __float2half(rand() / float(RAND_MAX));
    cudaMemset(d_c, 0, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float cpu_val = __half2float(h_c[i]);
        float gpu_val = __half2float(h_d_c[i]);
        float this_error = fabsf(gpu_val - cpu_val);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = fmaxf(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}


float testPerformance(
    void (*gpuSgemm) (half *, half *, half *, const int, const int, const int),
    dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(half);
    size_t size_b = K * N * sizeof(half);
    size_t size_c = M * N * sizeof(half);

    half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}