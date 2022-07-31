#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto kernel_implementation = R"""(
#ifndef __CUDACC_RTC__
#include <cufft.h>
#endif // __CUDACC_RTC__

__device__
extern void filter(float2 & value, int x, int y, int z);

// ZERO_MEAN
// RADIUS
// BLOCK_SIZE
// BLOCK_STEP
// IN_PLACE
// WARPS_PER_BLOCK
// WARP_SIZE
// TYPE
// SCALE
// PEAK (optional)

#if ZERO_MEAN
// __device__ const float window_freq[]; // frequency response of the window
#endif // ZERO_MEAN

__device__
static int calc_pad_size(int size, int block_size, int block_step) {
    return size + ((size % block_size) ? block_size - size % block_size : 0) + max(block_size - block_step, block_step) * 2;
}

__device__
static int calc_pad_num(int size, int block_size, int block_step) {
    return (calc_pad_size(size, block_size, block_step) - block_size) / block_step + 1;
}

__device__
static float to_float(TYPE x) {
    return static_cast<float>(x) * static_cast<float>(SCALE);
}

__device__
static TYPE from_float(float x) {
#ifdef PEAK
    x /= static_cast<float>(SCALE);
    x = fmaxf(0.0f, fminf(x + 0.5f, static_cast<float>(PEAK)));
    return static_cast<TYPE>(__float2int_rz(x));
#else // PEAK // only integral types define it
    return static_cast<TYPE>(x / static_cast<float>(SCALE));
#endif // PEAK
}

extern "C"
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
__global__
void im2col(
    // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, padded_block_size)
    float * __restrict__ dstp,
    const TYPE * __restrict__ srcp, // shape: (2*radius+1, vertical_size, horizontal_size)
    int width,
    int height
) {

    int radius = static_cast<int>(RADIUS);
    int block_size = static_cast<int>(BLOCK_SIZE);
    int padded_block_size = IN_PLACE ? (block_size / 2 + 1) * 2 : block_size;
    int block_step = static_cast<int>(BLOCK_STEP);

    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int num_blocks = vertical_num * horizontal_num;

    for (int i = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE; i < num_blocks; i += gridDim.x * WARPS_PER_BLOCK) {
        int ix = i % horizontal_num;
        int iy = i / horizontal_num;
        auto dst = &dstp[i * (2 * radius + 1) * block_size * padded_block_size];
        for (int j = 0; j < 2 * radius + 1; j++) {
            auto src = &srcp[(j * vertical_size + iy * block_step) * horizontal_size + ix * block_step];
            for (int k = threadIdx.x % WARP_SIZE; k < block_size * block_size; k += WARP_SIZE) {
                int kx = k % block_size;
                int ky = k / block_size;
                float val = to_float(src[ky * horizontal_size + kx]) * window[j * block_size * block_size + k];
#if IN_PLACE == 1
                dst[(j * block_size + k / block_size) * padded_block_size + k % block_size] = val;
#else
                dst[j * block_size * block_size + k] = val;
#endif
            }
        }
    }
}

extern "C"
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
__global__
void frequency_filtering(
    float2 * data,
    int num_blocks
) {

    int radius = static_cast<int>(RADIUS);
    int block_size_1d = static_cast<int>(BLOCK_SIZE);

    // each warp is responsible for a single block
    // assume that blockDim.x % WARP_SIZE == 0

    int block_size_x = block_size_1d / 2 + 1;
    int block_size_2d = block_size_1d * block_size_x;
    int block_size_3d = (2 * radius + 1) * block_size_2d;

    for (int i = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE; i < num_blocks; i += gridDim.x * WARPS_PER_BLOCK) {
#if ZERO_MEAN
        float gf;
        if (threadIdx.x % WARP_SIZE == 0) {
            gf = data[i * block_size_3d].x / window_freq[0];
        }
        gf = __shfl_sync(0xFFFFFFFF, gf, 0);
#endif // ZERO_MEAN

        for (int j = threadIdx.x % WARP_SIZE; j < block_size_3d; j += WARP_SIZE) {
            float2 local_data = data[i * block_size_3d + j];

#if ZERO_MEAN
            // remove mean
            float val1 = gf * window_freq[j * 2];
            float val2 = gf * window_freq[j * 2 + 1];
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
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
__global__
void col2im(
    TYPE * __restrict__ dst, // shape: (2*radius+1, vertical_size, horizontal_size)
    // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, padded_block_size)
    const float * __restrict__ src,
    int width,
    int height
) {

    int radius = static_cast<int>(RADIUS);
    int block_size = static_cast<int>(BLOCK_SIZE);
    int padded_block_size = IN_PLACE ? (block_size / 2 + 1) * 2 : block_size;
    int block_step = static_cast<int>(BLOCK_STEP);

    // each thread is responsible for a single pixel
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int pad_x = (horizontal_size - width) / 2;
    int pad_y = (vertical_size - height) / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y < pad_y || y >= pad_y + height || x < pad_x || x >= pad_x + width) {
        return ;
    }

    float sum {};

    int i1 = (y - block_size + block_step) / block_step; // i1 is implicitly greater than 0
    int i2 = min(y / block_step, vertical_num - 1);
    int j1 = (x - block_size + block_step) / block_step; // j1 is implicitly greater than 0
    int j2 = min(x / block_step, horizontal_num - 1);

    for (int i = i1; i <= i2; i++) {
        int offset_y = y - i * block_step;
        for (int j = j1; j <= j2; j++) {
            int offset_x = x - j * block_step;
            auto src_offset = (((i * horizontal_num + j) * (2 * radius + 1) + radius) * block_size + offset_y) * padded_block_size + offset_x;
            auto window_offset = (radius * block_size + offset_y) * block_size + offset_x;
            sum += src[src_offset] * window[window_offset];
        }
    }

    dst[(radius * vertical_size + y) * horizontal_size + x] = from_float(sum);
}
)""";

#endif // KERNEL_HPP
