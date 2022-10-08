#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <VSHelper.h>

#include "dfttest2_cpu.h"
#include "kernel.hpp"


typedef unsigned char Vec16uc __attribute__((__vector_size__(16), __aligned__(16)));
typedef unsigned char Vec16uc_u __attribute__((__vector_size__(16), __aligned__(1)));
typedef unsigned short Vec16us __attribute__((__vector_size__(32), __aligned__(32)));
typedef unsigned short Vec16us_u __attribute__((__vector_size__(32), __aligned__(1)));
typedef float Vec16f_u __attribute__((__vector_size__(64), __aligned__(1)));


static inline Vec16uc __attribute__((__always_inline__)) load_16uc(const unsigned char * p) {
    struct loadu {
        Vec16uc_u v;
    } __attribute__((__packed__, __may_alias__));

    return ((const struct loadu*) p)->v;
}


static inline Vec16us __attribute__((__always_inline__)) load_16us(const unsigned short * p) {
    struct loadu {
        Vec16us_u v;
    } __attribute__((__packed__, __may_alias__));

    return ((const struct loadu*) p)->v;
}


static inline Vec16f __attribute__((__always_inline__)) load_16f(const float * p) {
    struct loadu_16f {
        Vec16f_u v;
    } __attribute__((__packed__, __may_alias__));

    return ((const struct loadu_16f*) p)->v;
}


static inline void __attribute__((__always_inline__)) store_16f(float * p, Vec16f a) {
    struct storeu_ps {
        Vec16f_u v;
    } __attribute__((__packed__, __may_alias__));

    ((struct storeu_ps*) p)->v = a;
}


static inline int calc_pad_size(int size, int block_size, int block_step) {
    return (
        size
        + ((size % block_size) ? block_size - size % block_size : 0)
        + std::max(block_size - block_step, block_step) * 2
    );
}


static inline int calc_pad_num(int size, int block_size, int block_step) {
    return (calc_pad_size(size, block_size, block_step) - block_size) / block_step + 1;
}


template <typename T>
static inline void reflection_padding_impl(
    T * VS_RESTRICT dst, // shape: (pad_height, pad_width)
    const T * VS_RESTRICT src, // shape: (height, stride)
    int width, int height, int stride,
    int block_size, int block_step
) {

    int pad_width = calc_pad_size(width, block_size, block_step);
    int pad_height = calc_pad_size(height, block_size, block_step);

    int offset_y = (pad_height - height) / 2;
    int offset_x = (pad_width - width) / 2;

    vs_bitblt(
        &dst[offset_y * pad_width + offset_x], pad_width * sizeof(T),
        src, stride * sizeof(T),
        width * sizeof(T), height
    );

    // copy left and right regions
    for (int y = offset_y; y < offset_y + height; y++) {
        auto dst_line = &dst[y * pad_width];

        for (int x = 0; x < offset_x; x++) {
            dst_line[x] = dst_line[offset_x * 2 - x];
        }

        for (int x = offset_x + width; x < pad_width; x++) {
            dst_line[x] = dst_line[2 * (offset_x + width) - 2 - x];
        }
    }

    // copy top region
    for (int y = 0; y < offset_y; y++) {
        std::memcpy(
            &dst[y * pad_width],
            &dst[(offset_y * 2 - y) * pad_width],
            pad_width * sizeof(T)
        );
    }

    // copy bottom region
    for (int y = offset_y + height; y < pad_height; y++) {
        std::memcpy(
            &dst[y * pad_width],
            &dst[(2 * (offset_y + height) - 2 - y) * pad_width],
            pad_width * sizeof(T)
        );
    }
}


static inline void reflection_padding(
    uint8_t * VS_RESTRICT dst, // shape: (pad_height, pad_width)
    const uint8_t * VS_RESTRICT src, // shape: (height, stride)
    int width, int height, int stride,
    int block_size, int block_step,
    int bytes_per_sample
) {

    if (bytes_per_sample == 1) {
        reflection_padding_impl(
            static_cast<uint8_t *>(dst),
            static_cast<const uint8_t *>(src),
            width, height, stride,
            block_size, block_step
        );
    } else if (bytes_per_sample == 2) {
        reflection_padding_impl(
            reinterpret_cast<uint16_t *>(dst),
            reinterpret_cast<const uint16_t *>(src),
            width, height, stride,
            block_size, block_step
        );
    } else if (bytes_per_sample == 4) {
        reflection_padding_impl(
            reinterpret_cast<uint32_t *>(dst),
            reinterpret_cast<const uint32_t *>(src),
            width, height, stride,
            block_size, block_step
        );
    }
}


static inline void load_block(
    Vec16f * VS_RESTRICT block,
    const uint8_t * VS_RESTRICT shifted_src,
    int radius,
    int block_size,
    int block_step,
    int width,
    int height,
    const Vec16f * VS_RESTRICT window,
    int bits_per_sample
) {

    float scale = 1.0f / (1 << (bits_per_sample - 8));
    if (bits_per_sample == 32) {
        scale = 255.0f;
    }

    int bytes_per_sample = (bits_per_sample + 7) / 8;

    assert(block_size == 16);
    block_size = 16; // unsafe

    int offset_x = calc_pad_size(width, block_size, block_step);
    int offset_y = calc_pad_size(height, block_size, block_step);

    if (bytes_per_sample == 1) {
        for (int i = 0; i < 2 * radius + 1; i++) {
            for (int j = 0; j < block_size; j++) {
                auto vec_input = load_16uc((const uint8_t *) shifted_src + (i * offset_y + j) * offset_x);
                auto vec_input_f = __builtin_convertvector(vec_input, Vec16f);
                block[i * block_size * 2 + j] = scale * window[i * block_size + j] * vec_input_f;
            }
        }
    }
    if (bytes_per_sample == 2) {
        for (int i = 0; i < 2 * radius + 1; i++) {
            for (int j = 0; j < block_size; j++) {
                auto vec_input = load_16us((const uint16_t *) shifted_src + (i * offset_y + j) * offset_x);
                auto vec_input_f = __builtin_convertvector(vec_input, Vec16f);
                block[i * block_size * 2 + j] = scale * window[i * block_size + j] * vec_input_f;
            }
        }
    }
    if (bytes_per_sample == 4) {
        for (int i = 0; i < 2 * radius + 1; i++) {
            for (int j = 0; j < block_size; j++) {
                auto vec_input_f = load_16f((const float *) shifted_src + (i * offset_y + j) * offset_x);
                block[i * block_size * 2 + j] = scale * window[i * block_size + j] * vec_input_f;
            }
        }
    }
}


static inline void store_block(
    float * VS_RESTRICT shifted_dst,
    const Vec16f * VS_RESTRICT shifted_block,
    int block_size,
    int block_step,
    int width,
    int height,
    const Vec16f * VS_RESTRICT shifted_window
) {

    assert(block_size == 16);
    block_size = 16; // unsafe

    for (int i = 0; i < block_size; i++) {
        Vec16f acc = load_16f((const float *) shifted_dst + (i * calc_pad_size(width, block_size, block_step)));
        acc = FMA(shifted_block[i], shifted_window[i], acc);
        store_16f((float *) shifted_dst + (i * calc_pad_size(width, block_size, block_step)), acc);
    }
}


static inline void store_frame(
    uint8_t * VS_RESTRICT dst,
    const float * VS_RESTRICT shifted_src,
    int width,
    int height,
    int dst_stride,
    int src_stride,
    int bits_per_sample
) {

    float scale = 1.0f / (1 << (bits_per_sample - 8));
    if (bits_per_sample == 32) {
        scale = 255.0f;
    }

    int bytes_per_sample = (bits_per_sample + 7) / 8;
    int peak = (1 << bits_per_sample) - 1;

    if (bytes_per_sample == 1) {
        auto dstp = (uint8_t *) dst;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                auto clamped = std::clamp(static_cast<int>(shifted_src[y * src_stride + x] / scale + 0.5f), 0, peak);
                dstp[y * dst_stride + x] = static_cast<uint8_t>(clamped);
            }
        }
    }
    if (bytes_per_sample == 2) {
        auto dstp = (uint16_t *) dst;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                auto clamped = std::clamp(static_cast<int>(shifted_src[y * src_stride + x] / scale + 0.5f), 0, peak);
                dstp[y * dst_stride + x] = static_cast<uint16_t>(clamped);
            }
        }
    }
    if (bytes_per_sample == 4) {
        auto dstp = (float *) dst;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                dstp[y * dst_stride + x] = shifted_src[y * src_stride + x] / scale;
            }
        }
    }
}


const VSFrameRef * VS_CC
#ifndef HAS_DISPATCH
DFTTestGetFrame
#else // HAS_DISPATCH
DFTTEST_GETFRAME_NAME
#endif // HAS_DISPATCH
(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<DFTTestData *>(*instanceData);

    if (activationReason == arInitial) {
        int start = std::max(n - d->radius, 0);
        auto vi = vsapi->getVideoInfo(d->node);
        int end = std::min(n + d->radius, vi->numFrames - 1);
        for (int i = start; i <= end; i++) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        return nullptr;
    } else if (activationReason != arAllFramesReady) {
        return nullptr;
    }

    auto vi = vsapi->getVideoInfo(d->node);

    DFTTestThreadData thread_data;

    auto thread_id = std::this_thread::get_id();
    if (d->num_uninitialized_threads.load(std::memory_order_acquire) == 0) {
        const auto & const_data = d->thread_data;
        thread_data = const_data.at(thread_id);
    } else {
        bool initialized = true;

        d->thread_data_lock.lock_shared();
        try {
            const auto & const_data = d->thread_data;
            thread_data = const_data.at(thread_id);
        } catch (const std::out_of_range &) {
            initialized = false;
        }
        d->thread_data_lock.unlock_shared();

        if (!initialized) {
            auto padded_size = (
                (2 * d->radius + 1) *
                calc_pad_size(vi->height, d->block_size, d->block_step) *
                calc_pad_size(vi->width, d->block_size, d->block_step) *
                vi->format->bytesPerSample
            );

            thread_data.padded = static_cast<uint8_t *>(std::malloc(padded_size));
            thread_data.padded2 = static_cast<float *>(std::malloc(
                calc_pad_size(vi->height, d->block_size, d->block_step) *
                calc_pad_size(vi->width, d->block_size, d->block_step) *
                sizeof(float)
            ));

            {
                std::lock_guard _ { d->thread_data_lock };
                d->thread_data.emplace(thread_id, thread_data);
            }

            d->num_uninitialized_threads.fetch_sub(1, std::memory_order_release);
        }
    }

    std::vector<std::unique_ptr<const VSFrameRef, decltype(vsapi->freeFrame)>> src_frames;
    src_frames.reserve(2 * d->radius + 1);
    for (int i = n - d->radius; i <= n + d->radius; i++) {
        src_frames.emplace_back(
            vsapi->getFrameFilter(std::clamp(i, 0, vi->numFrames - 1), d->node, frameCtx),
            vsapi->freeFrame
        );
    }

    auto & src_center_frame = src_frames[d->radius];
    auto format = vsapi->getFrameFormat(src_center_frame.get());

    const VSFrameRef * fr[] {
        d->process[0] ? nullptr : src_center_frame.get(),
        d->process[1] ? nullptr : src_center_frame.get(),
        d->process[2] ? nullptr : src_center_frame.get()
    };
    const int pl[] { 0, 1, 2 };
    std::unique_ptr<VSFrameRef, decltype(vsapi->freeFrame)> dst_frame {
        vsapi->newVideoFrame2(format, vi->width, vi->height, fr, pl, src_center_frame.get(), core),
        vsapi->freeFrame
    };

    for (int plane = 0; plane < format->numPlanes; plane++) {
        if (!d->process[plane]) {
            continue;
        }

        int width = vsapi->getFrameWidth(src_center_frame.get(), plane);
        int height = vsapi->getFrameHeight(src_center_frame.get(), plane);
        int stride = vsapi->getStride(src_center_frame.get(), plane) / vi->format->bytesPerSample;

        int padded_size_spatial = (
            calc_pad_size(height, d->block_size, d->block_step) *
            calc_pad_size(width, d->block_size, d->block_step)
        );

        std::memset(thread_data.padded2, 0,
            calc_pad_size(height, d->block_size, d->block_step) *
            calc_pad_size(width, d->block_size, d->block_step) *
            sizeof(float)
        );

        for (int i = 0; i < 2 * d->radius + 1; i++) {
            auto srcp = vsapi->getReadPtr(src_frames[i].get(), plane);
            reflection_padding(
                &thread_data.padded[(i * padded_size_spatial) * vi->format->bytesPerSample],
                srcp,
                width, height, stride,
                d->block_size, d->block_step,
                vi->format->bytesPerSample
            );
        }

        for (int i = 0; i < calc_pad_num(height, d->block_size, d->block_step); i++) {
            for (int j = 0; j < calc_pad_num(width, d->block_size, d->block_step); j++) {
                assert(d->block_size == 16);
                constexpr int block_size = 16;

                Vec16f block[7 * block_size * 2];

                int offset_x = calc_pad_size(width, d->block_size, d->block_step);

                load_block(
                    block,
                    &thread_data.padded[(i * offset_x + j) * d->block_step * vi->format->bytesPerSample],
                    d->radius, d->block_size, d->block_step,
                    width, height,
                    reinterpret_cast<const Vec16f *>(d->window.get()),
                    vi->format->bitsPerSample
                );

                fused(
                    block,
                    reinterpret_cast<const Vec16f *>(d->sigma.get()),
                    d->sigma2,
                    d->pmin,
                    d->pmax,
                    d->filter_type,
                    d->zero_mean,
                    reinterpret_cast<const Vec16f *>(d->window_freq.get()),
                    d->radius
                );

                store_block(
                    &thread_data.padded2[(i * offset_x + j) * d->block_step],
                    &block[d->radius * block_size * 2],
                    block_size,
                    d->block_step,
                    width,
                    height,
                    reinterpret_cast<const Vec16f *>(&d->window[d->radius * block_size * 2 * 16])
                );
            }
        }

        int pad_width = calc_pad_size(width, d->block_size, d->block_step);
        int pad_height = calc_pad_size(height, d->block_size, d->block_step);
        int offset_y = (pad_height - height) / 2;
        int offset_x = (pad_width - width) / 2;

        auto dstp = vsapi->getWritePtr(dst_frame.get(), plane);
        store_frame(
            dstp,
            &thread_data.padded2[(offset_y * pad_width + offset_x)],
            width,
            height,
            stride,
            pad_width,
            vi->format->bitsPerSample
        );
    }

    return dst_frame.release();
}
