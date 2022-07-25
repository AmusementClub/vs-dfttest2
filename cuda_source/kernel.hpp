#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto kernel_implementation = R"""(
#ifndef __CUDACC_RTC__
#include <cufft.h>
#endif // __CUDACC_RTC__

__device__
extern void filter(float2 * value, int x, int y, int z, int id);

#ifdef ZERO_MEAN
// __device__ const float dftgc[];
// NUM_WARPS

extern "C"
__global__
void remove_mean(
    float * __restrict__ data, // re-interpret complex as float for coallesced access
    float * __restrict__ mean_patch,
    int num_blocks,
    int radius,
    int block_size
) {

    // each warp is responsible for a single block
    // assume that blockDim.x % warpSize == 0

    __shared__ float storage[WARPS_PER_BLOCK];
    for (int i = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / warpSize; i < num_blocks; i += gridDim.x * (blockDim.x / warpSize)) {
        if (threadIdx.x % warpSize == 0) {
            storage[threadIdx.x / warpSize] = data[i * (2 * radius + 1) * block_size * (block_size / 2 + 1) * 2] / dftgc[0];
        }
        __syncwarp();
        float gf = storage[threadIdx.x / warpSize];
        for (int j = threadIdx.x % warpSize; j < block_size * (block_size / 2 + 1) * 2; j += warpSize) {
            float val = gf * dftgc[j];
            mean_patch[i * (2 * radius + 1) * block_size * (block_size / 2 + 1) * 2 + j] = val;
            data[i * (2 * radius + 1) * block_size * (block_size / 2 + 1) * 2 + j] -= val;
        }
        __syncwarp();
    }
}

extern "C"
__global__
void add_mean(
    float * __restrict__ data, // re-interpret complex as float for coallesced access
    const float * __restrict__ mean_patch,
    int size
) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        data[i] += mean_patch[i];
    }
}
#endif // ZERO_MEAN

extern "C"
__global__
void frequency_filtering(
    float2 * data, 
    int size, 
    int block_size_1d
) {

    int block_size_x = block_size_1d / 2 + 1;
    int block_size_y = block_size_1d;
    int block_size_2d = block_size_y * block_size_x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        filter(&data[i], i % block_size_x, (i % block_size_2d) / block_size_x, 0, i / block_size_2d);
    }
}
)""";

#endif // KERNEL_HPP
