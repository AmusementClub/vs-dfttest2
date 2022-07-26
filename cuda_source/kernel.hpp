#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto kernel_implementation = R"""(
#ifndef __CUDACC_RTC__
#include <cufft.h>
#endif // __CUDACC_RTC__

__device__
extern void filter(float2 & value, int x, int y, int z);

// WARPS_PER_BLOCK
// ZERO_MEAN

#if ZERO_MEAN
// __device__ const float dftgc[];
#endif // ZERO_MEAN

__device__
static int calc_pad_size(int size, int block_size, int block_step) {
    return size + ((size % block_size) ? block_size - size % block_size : 0) + max(block_size - block_step, block_step) * 2;
}

__device__
static int calc_pad_num(int size, int block_size, int block_step) {
    return (calc_pad_size(size, block_size, block_step) - block_size) / block_step + 1;
}

extern "C"
__global__
void im2col(
    float * __restrict__ dstp, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    const float * __restrict__ srcp, // shape: (2*radius+1, vertical_size, horizontal_size)
    int width, int height,
    int radius,
    int block_size, int block_step
) {

    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int num_blocks = vertical_num * horizontal_num;

    for (int i = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / warpSize; i < num_blocks; i += gridDim.x * WARPS_PER_BLOCK) {
        int ix = i % horizontal_num;
        int iy = i / horizontal_num;
        auto dst = &dstp[i * (2 * radius + 1) * block_size * block_size];
        for (int j = 0; j < 2 * radius + 1; j++) {
            auto src = &srcp[(j * vertical_size + iy * block_step) * horizontal_size + ix * block_step];
            for (int k = threadIdx.x % warpSize; k < block_size * block_size; k += warpSize) {
                int kx = k % block_size;
                int ky = k / block_size;
                dst[j * block_size * block_size + k] = src[ky * horizontal_size + kx] * window[j * block_size * block_size + k];
            }
        }
    }
}

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

#if ZERO_MEAN
    __shared__ float storage[WARPS_PER_BLOCK];
#endif // ZERO_MEAN

    int block_size_x = block_size_1d / 2 + 1;
    int block_size_2d = block_size_1d * block_size_x;
    int block_size_3d = (2 * radius + 1) * block_size_2d;

    for (int i = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / warpSize; i < num_blocks; i += gridDim.x * WARPS_PER_BLOCK) {
#if ZERO_MEAN
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

#if ZERO_MEAN
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

#if ZERO_MEAN
            // add mean
            local_data.x += val1;
            local_data.y += val2;
#endif // ZERO_MEAN

            data[i * block_size_3d + j] = local_data;
        }
    }
}

extern "C"
__global__
void col2im(
    float * __restrict__ dst, // shape: (2*radius+1, vertical_size, horizontal_size)
    const float * __restrict__ src, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    int width, int height,
    int radius,
    int block_size, int block_step
) {

    // each thread is responsible for a single pixel
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int pad_x = (horizontal_size - width) / 2;
    int pad_y = (vertical_size - height) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < pad_y || y >= pad_y + height || x < pad_x || x >= pad_x + width) {
        return ;
    }

    float sum {};

    int i1 = (y - block_size + block_step) / block_step;
    int i2 = y / block_step;
    int j1 = (x - block_size + block_step) / block_step;
    int j2 = x / block_step;

    for (int i = i1; i <= i2; i++) {
        int offset_y = y - i * block_step;
        for (int j = j1; j <= j2; j++) {
            int offset_x = x - j * block_step;
            auto src_offset = (((i * horizontal_num + j) * (2 * radius + 1) + radius) * block_size + offset_y) * block_size + offset_x;
            auto window_offset = (radius * block_size + offset_y) * block_size + offset_x;
            sum += src[src_offset] * window[window_offset];
        }
    }

    dst[(radius * vertical_size + y) * horizontal_size + x] = sum;
}
)""";

#endif // KERNEL_HPP
