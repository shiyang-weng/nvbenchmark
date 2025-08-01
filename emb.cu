#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
// #include <cuda/std/bfloat16>
#include <device_launch_parameters.h>

#define EMBEDDING_DIM 256  // BLOCK_SIZE * FETCH_PER_WI == EMBEDDING_DIM
#define NUM_EMBEDDINGS 2000000
#define BATCH_SIZE 65536
#define BLOCK_SIZE 256
// #define WARP_SIZE 32
#define FETCH_PER_WI 1
#define NUM_ITERS 10
#define MULTIHOT_SIZE 10    // 每个block处理MULTIHOT_SIZE行

void check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " - " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}
#define CHECK_CUDA(call) check_cuda_error((call), __FILE__, __LINE__)

template<typename scalar_t>
__global__ void embedding_block_kernel(
    const scalar_t* __restrict__ weights,
    const int* __restrict__ indices,
    scalar_t* __restrict__ output) {

    int out_id = (blockIdx.x * blockDim.x * FETCH_PER_WI) + threadIdx.x;
    scalar_t sum[FETCH_PER_WI] = {0};

    for (int j = 0; j < MULTIHOT_SIZE; j++) {
        int index = indices[j + MULTIHOT_SIZE * blockIdx.x]; // 每个block处理MULTIHOT_SIZE行
        int weight_id = (index*EMBEDDING_DIM) + threadIdx.x;
        #pragma unroll
        for (int i = 0; i < FETCH_PER_WI; i++) {
            sum[i] += weights[weight_id + i*blockDim.x];
        }
    }

    for (int i = 0; i < FETCH_PER_WI; i++) {
        output[out_id + i*blockDim.x] = sum[i];
    }
}

template<typename scalar_t>
bool check_out(scalar_t* weights, int* h_indices, scalar_t* h_output, int batch_size) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            scalar_t refe = 0;
            for (int k = 0; k < MULTIHOT_SIZE; k++) {
                refe += weights[h_indices[k+MULTIHOT_SIZE*i]*EMBEDDING_DIM + j];
            }
            scalar_t test = h_output[i*EMBEDDING_DIM + j];
            if (test - refe > 1e-3 || test - refe < -1e-3) {
                std::cout << i << " " << j << " " << refe << " " << test << std::endl;
                return false;
            }
        }
    }
    return true;
}

void test_bandwidth() {
    using scalar_t = float;
    if (EMBEDDING_DIM != BLOCK_SIZE*FETCH_PER_WI) {
        return;
    }
    size_t table_size = NUM_EMBEDDINGS * EMBEDDING_DIM * sizeof(scalar_t);
    size_t output_size = BATCH_SIZE * EMBEDDING_DIM * sizeof(scalar_t);

    scalar_t *h_table = new scalar_t[NUM_EMBEDDINGS * EMBEDDING_DIM];
    int *h_indices = new int[BATCH_SIZE*MULTIHOT_SIZE];
    scalar_t *h_output = new scalar_t[BATCH_SIZE * EMBEDDING_DIM];

    // 初始化数据
    for (int i = 0; i < NUM_EMBEDDINGS * EMBEDDING_DIM; ++i) {
        h_table[i] = static_cast<scalar_t>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < BATCH_SIZE * MULTIHOT_SIZE; ++i) {
        scalar_t tmp = rand() % NUM_EMBEDDINGS; h_indices[i] = tmp > 0 ? tmp : -tmp;
        // h_indices[i] = (i+7) % NUM_EMBEDDINGS;
    }

    // 分配设备内存
    scalar_t *d_table, *d_output;
    int *d_indices;
    CHECK_CUDA(cudaMalloc(&d_table, table_size));
    CHECK_CUDA(cudaMalloc(&d_indices, MULTIHOT_SIZE * BATCH_SIZE * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, output_size));

    // 数据传输
    CHECK_CUDA(cudaMemcpy(d_table, h_table, table_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices, MULTIHOT_SIZE * BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // 预热
    for (int i = 0; i < 5; ++i) {
        embedding_block_kernel<<<BATCH_SIZE, BLOCK_SIZE>>>(d_table, d_indices, d_output);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // 带宽测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERS; ++i) {
        embedding_block_kernel<<<BATCH_SIZE, BLOCK_SIZE>>>(d_table, d_indices, d_output);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // 计算带宽
    float total_bytes = NUM_ITERS * BATCH_SIZE * (EMBEDDING_DIM * sizeof(scalar_t)) * (MULTIHOT_SIZE + 1);
    float bandwidth = (total_bytes / (elapsed_ms / 1000.0f)) / (1024 * 1024 * 1024);

    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));

    std::cout << "Total Bytes: " << total_bytes / (1024 * 1024 * 1024) << " GB" << std::endl;
    std::cout << "Global Memory Bandwidth: " << bandwidth << " GB/s" << std::endl;
    std::cout << "Execution Time: " << elapsed_ms / NUM_ITERS << " ms per batch" << std::endl;

    if (!check_out(h_table, h_indices, h_output, BATCH_SIZE)) {
        std::cout << "error\n";
    } else {
        std::cout << "accuracy pass\n";
    }

    // 清理资源
    delete[] h_table;
    delete[] h_indices;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_table));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main() {
    test_bandwidth();
    return 0;
}