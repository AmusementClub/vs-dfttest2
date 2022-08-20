#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto kernel_implementation = R"""(
__device__
extern void filter(float2 & value, int x, int y, int z);

// ZERO_MEAN
// RADIUS
// BLOCK_SIZE
// BLOCK_STEP
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
    x = fmaxf(0.0f, fminf(x, static_cast<float>(PEAK)));
    return static_cast<TYPE>(__float2int_rz(x + 0.5f));
#else // PEAK // only integral types define it
    return static_cast<TYPE>(x / static_cast<float>(SCALE));
#endif // PEAK
}

// im2col + rdft + frequency_filtering + irdft
extern "C"
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
__global__
void fused(
    float * __restrict__ dstp, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    const TYPE * __restrict__ srcp, // shape: (2*radius+1, vertical_size, horizontal_size)
    int width,
    int height
) {

    constexpr int radius = static_cast<int>(RADIUS);
    constexpr int block_size = static_cast<int>(BLOCK_SIZE);
    constexpr int block_step = static_cast<int>(BLOCK_STEP);

    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int num_blocks = vertical_num * horizontal_num;

    constexpr int warp_size = static_cast<int>(WARP_SIZE);
    constexpr int warps_per_block = static_cast<int>(WARPS_PER_BLOCK);
    constexpr int transpose_stride = (warp_size % block_size == 0) ? block_size + 1 : block_size;
    __shared__ float2 shared_transpose_buffer[warps_per_block * block_size * transpose_stride];

    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    auto transpose_buffer = &shared_transpose_buffer[warp_id * block_size * transpose_stride];

    for (int block_id = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE; block_id < num_blocks; block_id += gridDim.x * WARPS_PER_BLOCK) {
        int ix = block_id % horizontal_num;
        int iy = block_id / horizontal_num;

        if (lane_id < block_size) {
            constexpr int active_mask = (1 << block_size) - 1;
            float2 thread_data[(2 * radius + 1) * block_size];

            // im2col
            #pragma unroll
            for (int i = 0; i < 2 * radius + 1; i++) {
                auto src = &srcp[(i * vertical_size + iy * block_step) * horizontal_size + ix * block_step];
                auto local_thread_data = &thread_data[i * block_size];
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float *) local_thread_data)[j] = to_float(src[j * horizontal_size + lane_id]) * window[(i * block_size + j) * block_size + lane_id];
                }
            }

            // rdft
            #pragma unroll
            for (int i = 0; i < 2 * radius + 1; i++) {
                auto local_thread_data = &thread_data[i * block_size];

                __syncwarp(active_mask);
                // transpose store of real data
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float *) transpose_buffer)[j * transpose_stride + lane_id] = ((float *) local_thread_data)[j];
                }

                __syncwarp(active_mask);
                // transpose load of real data
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    ((float *) local_thread_data)[j] = ((float *) transpose_buffer)[lane_id * transpose_stride + j];
                }

                __syncwarp(active_mask);
                rdft<block_size>((float *) local_thread_data);

                // transpose store of complex data
                #pragma unroll
                for (int j = 0; j < block_size / 2 + 1; j++) {
                    transpose_buffer[lane_id * transpose_stride + j] = local_thread_data[j];
                }

                __syncwarp(active_mask);
                if (lane_id < block_size / 2 + 1) {
                    // transpose load of complex data
                    #pragma unroll
                    for (int j = 0; j < block_size; j++) {
                        local_thread_data[j] = transpose_buffer[j * transpose_stride + lane_id];
                    }

                    __syncwarp((1 << (block_size / 2 + 1)) - 1);
                    dft<block_size>((float *) local_thread_data);
                }
            }

            if (lane_id < block_size / 2 + 1) {
                #pragma unroll
                for (int i = 0; i < block_size; i++) {
                    dft<2 * radius + 1>((float *) &thread_data[i], block_size);
                }
            }

            // frequency_filtering
            if (lane_id < block_size / 2 + 1) {
#if ZERO_MEAN
                float gf;
                if (lane_id == 0) {
                    gf = thread_data[0].x / window_freq[0];
                }
                gf = __shfl_sync((1 << (block_size / 2 + 1)) - 1, gf, 0);
#endif // ZERO_MEAN
                #pragma unroll
                for (int i = 0; i < 2 * radius + 1; i++) {
                    #pragma unroll
                    for (int j = 0; j < block_size; j++) {
                        float2 local_data = thread_data[i * block_size + j];

#if ZERO_MEAN
                        // remove mean
                        float val1 = gf * window_freq[((i * block_size + j) * (block_size / 2 + 1) + lane_id) * 2];
                        float val2 = gf * window_freq[((i * block_size + j) * (block_size / 2 + 1) + lane_id) * 2 + 1];
                        local_data.x -= val1;
                        local_data.y -= val2;
#endif // ZERO_MEAN

                        filter(local_data, lane_id, j, i);

#if ZERO_MEAN
                        // add mean
                        local_data.x += val1;
                        local_data.y += val2;
#endif // ZERO_MEAN

                        thread_data[i * block_size + j] = local_data;
                    }
                }
            }

            // irdft
            if (lane_id < block_size / 2 + 1) {
                #pragma unroll
                for (int i = 0; i < block_size; i++) {
                    idft<2 * radius + 1>((float *) &thread_data[i], block_size);
                }
            }

            // this is not a full 3d irdft, because only a single slice is required
            auto local_thread_data = &thread_data[radius * block_size];

            if (lane_id < block_size / 2 + 1) {
                __syncwarp((1 << (block_size / 2 + 1)) - 1);
                idft<block_size>((float *) local_thread_data);

                // transpose store of complex data
                #pragma unroll
                for (int j = 0; j < block_size; j++) {
                    transpose_buffer[j * transpose_stride + lane_id] = local_thread_data[j];
                }
            }

            __syncwarp(active_mask);
            #pragma unroll
            for (int j = 0; j < block_size / 2 + 1; j++) {
                // transpose load of complex data
                local_thread_data[j].x = transpose_buffer[lane_id * transpose_stride + j].x;
                local_thread_data[j].y = transpose_buffer[lane_id * transpose_stride + j].y;
            }

            __syncwarp(active_mask);
            irdft<block_size>((float *) local_thread_data);

            #pragma unroll
            for (int j = 0; j < block_size; j++) {
                ((float *) transpose_buffer)[j * transpose_stride + lane_id] = ((float *) local_thread_data)[j == 0 ? j : block_size - j];
            }

            __syncwarp(active_mask);
            auto local_dst = &dstp[(block_id * (2 * radius + 1) + radius) * block_size * block_size];
            #pragma unroll
            for (int j = 0; j < block_size; j++) {
                local_dst[j * block_size + lane_id] = ((float *) transpose_buffer)[lane_id * transpose_stride + j];
            }
        }
    }
}

extern "C"
__launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE)
__global__
void col2im(
    TYPE * __restrict__ dst, // shape: (2*radius+1, vertical_size, horizontal_size)
    const float * __restrict__ src, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    int width,
    int height
) {

    int radius = static_cast<int>(RADIUS);
    int block_size = static_cast<int>(BLOCK_SIZE);
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
            auto src_offset = (((i * horizontal_num + j) * (2 * radius + 1) + radius) * block_size + offset_y) * block_size + offset_x;
            auto window_offset = (radius * block_size + offset_y) * block_size + offset_x;
            sum += src[src_offset] * window[window_offset];
        }
    }

    dst[(radius * vertical_size + y) * horizontal_size + x] = from_float(sum);
}
)""";

#endif // KERNEL_HPP
