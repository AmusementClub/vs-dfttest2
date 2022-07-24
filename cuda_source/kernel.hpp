#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto kernel_header_template = R"""(
#ifndef __CUDACC_RTC__
#include <cufft.h>
#endif // __CUDACC_RTC__

__device__
extern void filter(float2 * value, int x, int y, int z, int id);

extern "C"
__global__
void frequency_filtering(float2 * data, int size, int block_size_1d) {
    int block_size_x = block_size_1d / 2 + 1;
    int block_size_y = block_size_1d;
    int block_size_2d = block_size_y * block_size_x;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
        filter(&data[i], i % block_size_x, (i % block_size_2d) / block_size_x, 0, i / block_size_2d);
    }
}
)""";

#endif // KERNEL_HPP
