#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto kernel_implementation = R"""(
#ifndef __CUDACC_RTC__
#include <cufft.h>
#endif // __CUDACC_RTC__

__device__
extern void filter(float2 & value, int x, int y, int z);

// WARPS_PER_BLOCK

#ifdef ZERO_MEAN
// __device__ const float dftgc[];
#endif // ZERO_MEAN

extern "C"
__global__
void frequency_filtering(
    float2 * data,
    int num_blocks,
    int radius,
    int block_size_1d
) {

    // each warp is responsible for a single block
    // assume that blockDim.x % warpSize == 0

#ifdef ZERO_MEAN
    __shared__ float storage[WARPS_PER_BLOCK];
#endif // ZERO_MEAN

    int block_size_x = block_size_1d / 2 + 1;
    int block_size_2d = block_size_1d * block_size_x;
    int block_size_3d = (2 * radius + 1) * block_size_2d;

    for (int i = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / warpSize; i < num_blocks; i += gridDim.x * WARPS_PER_BLOCK) {
#ifdef ZERO_MEAN
        __syncwarp();
        if (threadIdx.x % warpSize == 0) {
            storage[threadIdx.x / warpSize] = data[i * block_size_3d].x / dftgc[0];
        }
        __syncwarp();
        float gf = storage[threadIdx.x / warpSize];
        __syncwarp();
#endif // ZERO_MEAN

        for (int j = threadIdx.x % warpSize; j < block_size_3d; j += warpSize) {
            float2 local_data = data[i * block_size_3d + j];

#ifdef ZERO_MEAN
            // remove mean
            float val1 = gf * dftgc[j * 2];
            float val2 = gf * dftgc[j * 2 + 1];
            local_data.x -= val1;
            local_data.y -= val2;
#endif // ZERO_MEAN

            filter(
                local_data,
                j % block_size_x,
                (j % block_size_2d) / block_size_x,
                (j % block_size_3d) / block_size_2d
            );

#ifdef ZERO_MEAN
            // add mean
            local_data.x += val1;
            local_data.y += val2;
#endif // ZERO_MEAN

            data[i * block_size_3d + j] = local_data;
        }
    }
}
)""";

#endif // KERNEL_HPP
