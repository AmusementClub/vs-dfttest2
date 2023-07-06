#ifndef DFTTEST2_CPU_H
#define DFTTEST2_CPU_H

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <unordered_map>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <vectorclass.h>


static inline void vs_aligned_free_float(float * ptr) {
    vs_aligned_free(static_cast<void *>(ptr));
}


struct DFTTestThreadData {
    uint8_t * padded; // shape: (pad_height, pad_width)
    float * padded2; // shape: (pad_height, pad_width)
};


struct DFTTestData {
    VSNodeRef * node;
    int radius;
    int block_size;
    int block_step;
    std::array<bool, 3> process;
    bool zero_mean;
    std::unique_ptr<float [], decltype(&vs_aligned_free_float)> window { nullptr, &vs_aligned_free_float };
    std::unique_ptr<float [], decltype(&vs_aligned_free_float)> window_freq { nullptr, &vs_aligned_free_float };
    std::unique_ptr<float [], decltype(&vs_aligned_free_float)> sigma { nullptr, &vs_aligned_free_float };
    int filter_type;
    float sigma2;
    float pmin;
    float pmax;

    std::atomic<int> num_uninitialized_threads;
    std::unordered_map<std::thread::id, DFTTestThreadData> thread_data;
    std::shared_mutex thread_data_lock;
};

#if defined HAS_DISPATCH
#include <cpu_dispatch.h>
#else // HAS_DISPATCH
extern const VSFrameRef *VS_CC DFTTestGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept;

extern bool supported_arch() noexcept;

extern const char * target_arch() noexcept;
#endif // HAS_DISPATCH

#endif // DFTTEST2_CPU_H
