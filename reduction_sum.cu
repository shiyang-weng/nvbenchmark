#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_WARMUP 100
#define NUM_ITER 100
#define WARP_SIZE 32
#define FETCH_PER_WI 32
#define T_SIZE (1024 * 1024 * 512) // 2GB数据量

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ float blockReduceSum(float val) {
    __shared__ float res_per_warp[32];
    __shared__ float res;

    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) {
        res_per_warp[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = warpReduceSum(val);
        if (lane == 0) {
            res = val;
        }
    }
    __syncthreads();

    return res;
}


__global__ void global_bandwidth_v1_warp_offset(float* A, float* B, float* C) {
    // 假设
    //      每个warp     处理一行1024个连续input

    // 那么
    //      每个thread   处理 FETCH_PER_WI=1024/32=32 个不连续的input
    //      thread总数   T_size/FETCH_PER_WI
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id / WARP_SIZE;
    int warp_start = warp_id * 1024;
    int warp_thread_id = threadIdx.x % WARP_SIZE;

    int id = warp_start + warp_thread_id;
    float sum = 0;

    // #pragma unroll
    for(int i=0; i<FETCH_PER_WI; i++) {
        sum += A[id];
        id += 32;
    }

    // TODO: warp reduction proformance
    sum = warpReduceSum(sum);
    if (warp_thread_id == 0) {
        B[blockIdx.x * (blockDim.x/32) + threadIdx.x/32] = sum;
    }

    id = warp_start + warp_thread_id;
    for(int i=0; i<FETCH_PER_WI; i++) {
        C[id] = A[id] - sum;
        id += 32;
    }
}

__global__ void global_bandwidth_v1_warp_offset_cached(float* A, float* B, float* C) {
    // 假设
    //      每个warp     处理一行1024个连续input

    // 那么
    //      每个thread   处理 FETCH_PER_WI=1024/32=32 个不连续的input
    //      thread总数   T_size/FETCH_PER_WI
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id / WARP_SIZE;
    int warp_start = warp_id * 1024;
    int warp_thread_id = threadIdx.x % WARP_SIZE;

    int id = warp_start + warp_thread_id;
    float sum = 0;
    float tmp[FETCH_PER_WI];

    #pragma unroll
    for(int i=0; i<FETCH_PER_WI; i++) {
        tmp[i] = A[id];
        sum += tmp[i];
        id += 32;
    }

    sum = warpReduceSum(sum);

    id = warp_start + warp_thread_id;
    for(int i=0; i<FETCH_PER_WI; i++) {
        C[id] = tmp[i] - sum;
        id += 32;
    }
}

int main() {
    auto test_func = global_bandwidth_v1_warp_offset_cached;
    size_t vec_size = 1;
    float *d_A, *d_B, *d_C;
    size_t data_size = T_SIZE * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_A, data_size);
    cudaMalloc(&d_B, data_size / FETCH_PER_WI);
    cudaMalloc(&d_C, data_size);

    // 配置内核参数
    dim3 blockDim(256);
    dim3 gridDim((T_SIZE / FETCH_PER_WI + blockDim.x - 1) / blockDim.x  / vec_size);

    for (int i = 0; i < NUM_WARMUP; i++) {
        test_func<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    }

    cudaDeviceSynchronize();
    // 执行测试
    clock_t e2e_start, e2e_end;
    e2e_start = clock();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITER; i++) {
        test_func<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaDeviceSynchronize();
    e2e_end = clock();

    // 计算带宽
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= NUM_ITER;
    double e2e_time = ((double)e2e_end-e2e_start)/CLOCKS_PER_SEC * 1000 / NUM_ITER;
    double bandwidth = (data_size / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0) * (2*FETCH_PER_WI + 1) / FETCH_PER_WI;

    printf("测试数据量: %.2f GB\n", data_size / (1024.0 * 1024.0 * 1024.0));
    printf("执行时间: %.3f ms\n", milliseconds);
    printf("E2E时间: %.3f ms\n", e2e_time);
    printf("内存带宽: %.2f GB/s\n", bandwidth);

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}