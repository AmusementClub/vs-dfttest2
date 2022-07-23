#ifndef KERNEL_HPP
#define KERNEL_HPP

static const auto kernel_header_template = R"""(
#ifndef __CUDACC_RTC__
#include <cufft.h>
#endif // __CUDACC_RTC__

__device__
extern void filter(float2 * value, int x, int y, int id);

extern "C"
__global__
void frequency_filtering(float2 * data, int n, int block_size) {
    int sq_block_size = block_size * block_size;

    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < n; x += blockDim.x * gridDim.x) {
        filter(&data[x], x % block_size, x % sq_block_size / block_size, x / sq_block_size);
    }
}
)""";

#endif // KERNEL_HPP
