#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define NUM_WARMUP 100
#define NUM_ITER 100
#define WARP_SIZE 32
#define FETCH_PER_WI 8
// #define T_SIZE (1024 * 1024 * 512) // 2GB数据量
#define T_SIZE (1024 * 864 * 4) // 2GB数据量

__global__ void global_bandwidth_v1_local_offset(float* A, float* B) {
    // 一共读T_size
    // 每个thread   读取 FETCH_PER_WI 个元素; stride 256;
    // 每个block    读取 256*FETCH_PER_WI 个连续元素;
    // GridDim T_size / (256*FETCH_PER_WI)
    // thread总数 2G / FETCH_PER_WI
    // 108 SM * 64warp * 32 = 221184个线程
    int id = (blockIdx.x * blockDim.x * FETCH_PER_WI) + threadIdx.x;
    float sum = 0;

    #pragma unroll
    for(int i=0; i<FETCH_PER_WI; i++) {
        sum += A[id]; id += blockDim.x;
    }

    B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void global_bandwidth_v1_global_offset(float* A, float* B) {
    // 每个thread   读取 FETCH_PER_WI 个元素，stride global_size;
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_size = gridDim.x * blockDim.x;
    float sum = 0;

    for(int i=0; i<FETCH_PER_WI/4; i++) {
        sum += A[id]; id += global_size;
        sum += A[id]; id += global_size;
        sum += A[id]; id += global_size;
        sum += A[id]; id += global_size;
    }

    B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void global_bandwidth_v4_local_offset(float* A, float* B) {
    // 每个SM 65536 个32位寄存器
    // 每个SM 最多64warp，也就是每个warp能用1024寄存器，每个thread能用32个32位寄存器
    // 每个SM 8个block

    // 一共读T_size
    // 每个thread   读取 FETCH_PER_WI 个float4元素; stride 256;
    // 每个block    读取 256*FETCH_PER_WI 个连续float4元素;
    // GridDim T_size / (256*4*FETCH_PER_WI)
    int id = (blockIdx.x * blockDim.x * FETCH_PER_WI) + threadIdx.x;
    float4 sum = {0,0,0,0};

    #pragma unroll
    for(int i=0; i<FETCH_PER_WI; i++) {
        float4 val = reinterpret_cast<float4*>(A)[id];
        sum.x += val.x;
        sum.y += val.y;
        sum.z += val.z;
        sum.w += val.w;
        id += blockDim.x;
    }

    B[blockIdx.x * blockDim.x + threadIdx.x] = sum.x + sum.y + sum.z + sum.w;
}

__global__ void global_bandwidth_v1_warp_offset(float* A, float* B) {
    // 假设
    //      每个warp     32thread，处理一行1024个连续input

    // 那么
    //      每个thread   处理FETCH_PER_WI=1024/32=32个不连续的input
    //      thread总数   T_size/FETCH_PER_WI个
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_id / WARP_SIZE;
    int warp_start = warp_id * 1024;
    int warp_thread_id = threadIdx.x % WARP_SIZE;

    int id = warp_start + warp_thread_id;
    float sum = 0;

    #pragma unroll
    for(int i=0; i<FETCH_PER_WI; i++) {
        sum += A[id];
        id += 32;
    }

    B[global_id] = sum;
}

int main() {
    auto test_func = global_bandwidth_v1_local_offset;
    size_t vec_size = 1;
    float *d_A, *d_B;
    size_t data_size = T_SIZE * sizeof(float);

    // 分配设备内存
    cudaMalloc(&d_A, data_size);
    cudaMalloc(&d_B, data_size / FETCH_PER_WI);

    // 配置内核参数
    dim3 blockDim(128);
    dim3 gridDim((T_SIZE / FETCH_PER_WI + blockDim.x - 1) / blockDim.x  / vec_size);

    for (int i = 0; i < NUM_WARMUP; i++) {
        test_func<<<gridDim, blockDim>>>(d_A, d_B);
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
        test_func<<<gridDim, blockDim>>>(d_A, d_B);
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
    double bandwidth = (data_size / (1024.0 * 1024.0 * 1024.0)) / (milliseconds / 1000.0) * (FETCH_PER_WI + 1) / FETCH_PER_WI;

    printf("测试数据量: %.2f GB\n", data_size / (1024.0 * 1024.0 * 1024.0));
    printf("执行时间: %.3f ms\n", milliseconds);
    printf("E2E时间: %.3f ms\n", e2e_time);
    printf("内存带宽: %.2f GB/s\n", bandwidth);

    // 清理资源
    cudaFree(d_A);
    cudaFree(d_B);
    return 0;
}