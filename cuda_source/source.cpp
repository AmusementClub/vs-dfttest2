#include <array>
#include <atomic>
#include <algorithm>
#include <concepts>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <VapourSynth.h>
#include <VSHelper.h>

#include <cuda.h>
#include <cufft.h>
#include <nvrtc.h>

#include "kernel.hpp"

static const char * cufftGetErrorString(cufftResult_t result) {
    switch (result) {
        case CUFFT_SUCCESS:
            return "success";
        case CUFFT_INVALID_PLAN:
            return "invalid plan handle";
        case CUFFT_ALLOC_FAILED:
            return "failed to allocate memory";
        case CUFFT_INVALID_VALUE:
            return "invalid value";
        case CUFFT_INTERNAL_ERROR:
            return "internal error";
        case CUFFT_EXEC_FAILED:
            return "execution failed";
        case CUFFT_SETUP_FAILED:
            return "the cuFFT library failed to initialize";
        case CUFFT_INVALID_SIZE:
            return "invalid transform size";
        case CUFFT_INCOMPLETE_PARAMETER_LIST:
            return "missing parameters in call";
        case CUFFT_INVALID_DEVICE:
            return "invalid device: execution of a plan was on different GPU than plan creation";
        case CUFFT_PARSE_ERROR:
            return "internal plan database error";
        case CUFFT_NO_WORKSPACE:
            return "no workspace has been provided prior to plan execution";
        case CUFFT_NOT_IMPLEMENTED:
            return "functionality not implemented";
        case CUFFT_NOT_SUPPORTED:
            return "operation not supported";
        default:
            return "unknown";
    }
}

static bool success(CUresult result) {
    return result == CUDA_SUCCESS;
}
static bool success(cufftResult_t result) {
    return result == CUFFT_SUCCESS;
}
static bool success(nvrtcResult result) {
    return result == NVRTC_SUCCESS;
}

#define showError(expr) show_error_impl(expr, # expr, __LINE__)
static void show_error_impl(CUresult result, const char * source, int line_no) {
    if (!success(result)) [[unlikely]] {
        const char * error_message;
        cuGetErrorString(result, &error_message);
        fprintf(stderr, "[%d] %s failed: %s\n", line_no, source, error_message);
    }
}
static void show_error_impl(cufftResult_t result, const char * source, int line_no) {
    if (!success(result)) [[unlikely]] {
        fprintf(stderr, "[%d] %s failed: %s\n", line_no, source, cufftGetErrorString(result));
    }
}
static void show_error_impl(nvrtcResult result, const char * source, int line_no) {
    if (!success(result)) [[unlikely]] {
        fprintf(stderr, "[%d] %s failed: %s\n", line_no, source, nvrtcGetErrorString(result));
    }
}

static void cuStreamDestroyCustom(CUstream stream) {
    showError(cuStreamDestroy(stream));
}

static void cuMemFreeCustom(CUdeviceptr p) {
    showError(cuMemFree(p));
}

static void cuMemFreeHostCustom(void * p) {
    showError(cuMemFreeHost(p));
}

static void cuModuleUnloadCustom(CUmodule module) {
    showError(cuModuleUnload(module));
}

static void cufftDestroyCustom(cufftHandle handle) {
    showError(cufftDestroy(handle));
}

static void nvrtcDestroyProgramCustom(nvrtcProgram * program) {
    showError(nvrtcDestroyProgram(program));
}

struct context_releaser {
    bool * context_retained {};
    CUdevice device;
    void release() {
        context_retained = nullptr;
    }
    ~context_releaser() {
        if (context_retained && *context_retained) {
            showError(cuDevicePrimaryCtxRelease(device));
        }
    }
};

struct context_popper {
    bool * context_pushed {};
    ~context_popper() {
        if (!context_pushed || *context_pushed) {
            showError(cuCtxPopCurrent(nullptr));
        }
    }
};

struct node_freer {
    const VSAPI * & vsapi;
    VSNodeRef * node {};
    void release() {
        node = nullptr;
    }
    ~node_freer() {
        if (node) {
            vsapi->freeNode(node);
        }
    }
};

template <typename T, auto deleter, bool unsafe=false>
    requires
        std::default_initializable<T> &&
        std::is_trivially_copy_assignable_v<T> &&
        std::convertible_to<T, bool> &&
        std::invocable<decltype(deleter), T> &&
        (std::is_pointer_v<T> || unsafe) // CUdeviceptr is not a pointer
struct Resource {
    T data;

    [[nodiscard]] constexpr Resource() noexcept = default;

    [[nodiscard]] constexpr Resource(T x) noexcept : data(x) {}

    [[nodiscard]] constexpr Resource(Resource&& other) noexcept
            : data(std::exchange(other.data, T{}))
    { }

    Resource& operator=(Resource&& other) noexcept {
        if (this == &other) return *this;
        deleter_(data);
        data = std::exchange(other.data, T{});
        return *this;
    }

    Resource operator=(Resource other) = delete;

    Resource(const Resource& other) = delete;

    constexpr operator T() const noexcept {
        return data;
    }

    constexpr auto deleter_(T x) noexcept {
        if (x) {
            deleter(x);
            x = T{};
        }
    }

    Resource& operator=(T x) noexcept {
        deleter_(data);
        data = x;
        return *this;
    }

    constexpr ~Resource() noexcept {
        deleter_(data);
    }
};

template <typename T>
static T square(const T & x) {
    return x * x;
}

static int calc_pad_size(int size, int block_size, int block_step) {
    return size + ((size % block_size) ? block_size - size % block_size : 0) + std::max(block_size - block_step, block_step) * 2;
}

static int calc_pad_num(int size, int block_size, int block_step) {
    return (calc_pad_size(size, block_size, block_step) - block_size) / block_step + 1;
}

static void reflection_padding(
    float * VS_RESTRICT dst, // shape: (pad_height, pad_width)
    const float * VS_RESTRICT src, // shape: (height, stride)
    int width, int height, int stride,
    int block_size, int block_step
) {

    int pad_width = calc_pad_size(width, block_size, block_step);
    int pad_height = calc_pad_size(height, block_size, block_step);

    int offset_y = (pad_height - height) / 2;
    int offset_x = (pad_width - width) / 2;

    vs_bitblt(
        &dst[offset_y * pad_width + offset_x], pad_width * sizeof(float),
        src, stride * sizeof(float),
        width * sizeof(float), height
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
        std::memcpy(&dst[y * pad_width], &dst[(offset_y * 2 - y) * pad_width], pad_width * sizeof(float));
    }

    // copy bottom region
    for (int y = offset_y + height; y < pad_height; y++) {
        std::memcpy(&dst[y * pad_width], &dst[(2 * (offset_y + height) - 2 - y) * pad_width], pad_width * sizeof(float));
    }
}

static void im2col(
    float * VS_RESTRICT dstp, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    const float * VS_RESTRICT srcp, // shape: (vertical_size, horizontal_size)
    const float * VS_RESTRICT window, // shape: (2*radius+1, block_size, block_size)
    int width, int height,
    int radius, int temporal_id,
    int block_size, int block_step
) {

    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);
    int horizontal_size = calc_pad_size(width, block_size, block_step);

    for (int i = 0; i < vertical_num; i++) {
        for (int j = 0; j < horizontal_num; j++) {
            auto src = &srcp[i * block_step * horizontal_size + j * block_step];
            for (int k = 0; k < block_size; k++) {
                for (int l = 0; l < block_size; l++) {
                    dstp[k * block_size + l] = src[k * horizontal_size + l] * window[(temporal_id * block_size + k) * block_size + l];
                }
            }
            dstp += (2 * radius + 1) * square(block_size);
        }
    }
}

static void col2im(
    float * VS_RESTRICT dst, // shape: (vertical_size, horizontal_size)
    const float * VS_RESTRICT src, // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    const float * VS_RESTRICT window, // shape: (2*radius+1, block_size, block_size)
    int width, int height,
    int radius,
    int block_size, int block_step
) {

    int horizontal_size = calc_pad_size(width, block_size, block_step);
    int horizontal_num = calc_pad_num(width, block_size, block_step);
    int vertical_size = calc_pad_size(height, block_size, block_step);
    int vertical_num = calc_pad_num(height, block_size, block_step);

    std::memset(dst, 0, vertical_size * horizontal_size * sizeof(float));
    for (int i = 0; i < vertical_num; i++) {
        for (int j = 0; j < horizontal_num; j++) {
            for (int k = 0; k < block_size; k++) {
                for (int l = 0; l < block_size; l++) {
                    dst[(i * block_step + k) * horizontal_size + j * block_step + l] +=
                        src[(((i * horizontal_num + j) * (2 * radius + 1) + radius) * block_size + k) * block_size + l] *
                        window[(radius * block_size + k) * block_size + l];
                }
            }
        }
    }
}

static std::variant<CUmodule, std::string> compile(
    const char * user_kernel,
    CUdevice device
) {

    int major;
    if (auto result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device); !success(result)) {
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        return std::string{error_message};
    }
    int minor;
    if (auto result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device); !success(result)) {
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        return std::string{error_message};
    }
    int compute_capability = major * 10 + minor;

    // find maximum supported architecture
    int num_archs;
    if (auto result = nvrtcGetNumSupportedArchs(&num_archs); !success(result)) {
        return std::string{nvrtcGetErrorString(result)};
    }
    const auto supported_archs = std::make_unique<int []>(num_archs);
    if (auto result = nvrtcGetSupportedArchs(supported_archs.get()); !success(result)) {
        return std::string{nvrtcGetErrorString(result)};
    }
    bool generate_cubin = compute_capability <= supported_archs[num_archs - 1];

    std::ostringstream kernel_source;
    kernel_source << "#define WARPS_PER_BLOCK 4\n";
    kernel_source << user_kernel;
    kernel_source << kernel_implementation;

    nvrtcProgram program;
    if (auto result = nvrtcCreateProgram(&program, kernel_source.str().c_str(), nullptr, 0, nullptr, nullptr); result != NVRTC_SUCCESS) {
        return std::string{nvrtcGetErrorString(result)};
    }
    Resource<nvrtcProgram *, nvrtcDestroyProgramCustom> destroyer { &program };

    const std::string arch_str = {
        generate_cubin ?
        "-arch=sm_" + std::to_string(compute_capability) :
        "-arch=compute_" + std::to_string(supported_archs[num_archs - 1])
    };

    const char * opts[] = {
        arch_str.c_str(),
        "-use_fast_math",
        "-std=c++17",
        "-modify-stack-limit=false"
    };

    if (nvrtcCompileProgram(program, static_cast<int>(std::extent_v<decltype(opts)>), opts) == NVRTC_SUCCESS) {
        size_t log_size;
        showError(nvrtcGetProgramLogSize(program, &log_size));
        if (log_size > 1) {
            std::string error_message;
            error_message.resize(log_size);
            showError(nvrtcGetProgramLog(program, error_message.data()));
            std::fprintf(stderr, "nvrtc: %s\n", error_message.c_str());
        }
    } else {
        size_t log_size;
        showError(nvrtcGetProgramLogSize(program, &log_size));
        std::string error_message;
        error_message.resize(log_size);
        showError(nvrtcGetProgramLog(program, error_message.data()));
        return error_message;
    }

    size_t cubin_size;
    if (auto result = nvrtcGetCUBINSize(program, &cubin_size); !success(result)) {
        return std::string{nvrtcGetErrorString(result)};
    }
    auto image = std::make_unique<char[]>(cubin_size);
    if (auto result = nvrtcGetCUBIN(program, image.get()); !success(result)) {
        return std::string{nvrtcGetErrorString(result)};
    }

    CUmodule module;
    if (auto result = cuModuleLoadData(&module, image.get()); !success(result)) {
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        return std::string{error_message};
    }

    return module;
}


struct DFTTestThreadData {
    float * padded; // shape: (pad_height, pad_width)
    float * h_spatial;
};


struct DFTTestData {
    VSNodeRef * node;
    std::unique_ptr<float []> window;
    int radius;
    int block_size;
    int block_step;
    CUdevice device; // device_id

    CUcontext context; // use primary stream for interoperability
    Resource<CUstream, cuStreamDestroyCustom> stream;
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_spatial; // shape: (pad_height, pad_width)
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_frequency; // (vertical_num, horizontal_num, block_size, block_size/2+1)
    std::mutex lock; // TODO: replace by `num_streams`

    Resource<cufftHandle, cufftDestroyCustom, true> rfft_handle; // 2-D or 3-D, depends on radius
    Resource<cufftHandle, cufftDestroyCustom, true> irfft_handle;

    Resource<CUmodule, cuModuleUnloadCustom> module;
    CUfunction filter_kernel;
    int filter_num_blocks;

    std::atomic<int> num_uninitialized_threads;
    std::unordered_map<std::thread::id, DFTTestThreadData> thread_data;
    std::shared_mutex thread_data_lock;
};

static void VS_CC DFTTestInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<const DFTTestData *>(*instanceData);

    auto vi = vsapi->getVideoInfo(d->node);
    vsapi->setVideoInfo(vi, 1, node);
}

static const VSFrameRef *VS_CC DFTTestGetFrame(
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
    } else if (activationReason == arAllFramesReady) {
        if (auto result = cuCtxPushCurrent(d->context); !success(result)) {
            std::ostringstream message;
            const char * error_message;
            showError(cuGetErrorString(result, &error_message));
            message << '[' << __LINE__ << "] cuCtxPushCurrent(): " << error_message;
            vsapi->setFilterError(message.str().c_str(), frameCtx);
            return nullptr;
        }

        context_popper context_popper;

        auto vi = vsapi->getVideoInfo(d->node);

        DFTTestThreadData thread_data;

        auto thread_id = std::this_thread::get_id();
        if (d->num_uninitialized_threads.load(std::memory_order_acquire) == 0) {
            const auto & const_padded = d->thread_data;
            thread_data = const_padded.at(thread_id);
        } else {
            bool initialized = true;

            d->thread_data_lock.lock_shared();
            try {
                thread_data = d->thread_data.at(thread_id);
            } catch (const std::out_of_range &) {
                initialized = false;
            }
            d->thread_data_lock.unlock_shared();

            if (!initialized) {
                auto padded_size = (
                    calc_pad_size(vi->height, d->block_size, d->block_step) *
                    calc_pad_size(vi->width, d->block_size, d->block_step) *
                    sizeof(float)
                );

                if (thread_data.padded = reinterpret_cast<float *>(std::malloc(padded_size)); !thread_data.padded) {
                    std::ostringstream message;
                    message << '[' << __LINE__ << "] malloc(h_padded) error";
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }

                auto spatial_size = (
                    (2 * d->radius + 1) *
                    calc_pad_num(vi->height, d->block_size, d->block_step) *
                    calc_pad_num(vi->width, d->block_size, d->block_step) *
                    square(d->block_size) *
                    sizeof(float)
                );

                if (auto result = cuMemHostAlloc(reinterpret_cast<void **>(&thread_data.h_spatial), spatial_size, 0); !success(result)) {
                    std::free(thread_data.padded);
                    std::ostringstream message;
                    const char * error_message;
                    showError(cuGetErrorString(result, &error_message));
                    message << '[' << __LINE__ << "] cuMemHostAlloc(): " << error_message;
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }

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

        std::unique_ptr<VSFrameRef, decltype(vsapi->freeFrame)> dst_frame {
            vsapi->newVideoFrame(format, vi->width, vi->height, src_center_frame.get(), core),
            vsapi->freeFrame
        };

        for (int plane = 0; plane < format->numPlanes; plane++) {
            int width = vsapi->getFrameWidth(src_center_frame.get(), plane);
            int height = vsapi->getFrameHeight(src_center_frame.get(), plane);
            int stride = vsapi->getStride(src_center_frame.get(), plane) / sizeof(float);

            for (int i = 0; i < 2 * d->radius + 1; i++) {
                auto srcp = vsapi->getReadPtr(src_frames[i].get(), plane);
                reflection_padding(
                    thread_data.padded,
                    reinterpret_cast<const float *>(srcp),
                    width, height, stride,
                    d->block_size, d->block_step
                );
                im2col(&thread_data.h_spatial[i * square(d->block_size)], thread_data.padded, d->window.get(), width, height, d->radius, i, d->block_size, d->block_step);
            }

            {
                std::lock_guard lock { d->lock };

                int spatial_size_bytes = (2 * d->radius + 1) * calc_pad_num(height, d->block_size, d->block_step) * calc_pad_num(width, d->block_size, d->block_step) * square(d->block_size) * sizeof(float);
                if (auto result = cuMemcpyHtoDAsync(d->d_spatial, thread_data.h_spatial, spatial_size_bytes, d->stream); !success(result)) {
                    std::ostringstream message;
                    const char * error_message;
                    showError(cuGetErrorString(result, &error_message));
                    message << '[' << __LINE__ << "] cuMemcpyHtoDAsync(): " << error_message;
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
                if (auto result = cufftExecR2C(d->rfft_handle, reinterpret_cast<cufftReal *>(d->d_spatial.data), reinterpret_cast<cufftComplex *>(d->d_frequency.data)); !success(result)) {
                    std::ostringstream message;
                    message << '[' << __LINE__ << "] cufft(rfft): " << cufftGetErrorString(result);
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
                {
                    int num_blocks = calc_pad_num(height, d->block_size, d->block_step) * calc_pad_num(width, d->block_size, d->block_step);
                    void * params[] { &d->d_frequency, &num_blocks, &d->radius, &d->block_size };
                    if (auto result = cuLaunchKernel(d->filter_kernel, static_cast<unsigned int>(d->filter_num_blocks), 1, 1, 128, 1, 1, 0, d->stream, params, nullptr); !success(result)) {
                        std::ostringstream message;
                        const char * error_message;
                        showError(cuGetErrorString(result, &error_message));
                        message << '[' << __LINE__ << "] cuLaunchKernel(frequency_filtering): " << error_message;
                        vsapi->setFilterError(message.str().c_str(), frameCtx);
                        return nullptr;
                    }
                }
                if (auto result = cufftExecC2R(d->irfft_handle, reinterpret_cast<cufftComplex *>(d->d_frequency.data), reinterpret_cast<cufftReal *>(d->d_spatial.data)); !success(result)) {
                    std::ostringstream message;
                    message << '[' << __LINE__ << "] cufft(irfft): " << cufftGetErrorString(result);
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
                if (auto result = cuMemcpyDtoHAsync(thread_data.h_spatial, d->d_spatial, spatial_size_bytes, d->stream); !success(result)) {
                    std::ostringstream message;
                    const char * error_message;
                    showError(cuGetErrorString(result, &error_message));
                    message << '[' << __LINE__ << "] cuMemcpyDtoHAsync(): " << error_message;
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
                if (auto result = cuStreamSynchronize(d->stream); !success(result)) {
                    std::ostringstream message;
                    const char * error_message;
                    showError(cuGetErrorString(result, &error_message));
                    message << '[' << __LINE__ << "] cuStreamSynchronize(): " << error_message;
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
            }

            col2im(thread_data.padded, thread_data.h_spatial, d->window.get(), width, height, d->radius, d->block_size, d->block_step);

            int pad_width = calc_pad_size(width, d->block_size, d->block_step);
            int pad_height = calc_pad_size(height, d->block_size, d->block_step);
            int offset_y = (pad_height - height) / 2;
            int offset_x = (pad_width - width) / 2;

            auto dstp = vsapi->getWritePtr(dst_frame.get(), plane);
            vs_bitblt(
                dstp, stride * sizeof(float),
                &thread_data.padded[offset_y * pad_width + offset_x], pad_width * sizeof(float),
                width * sizeof(float), height
            );
        }

        return dst_frame.release();
    }

    return nullptr;
}

static void VS_CC DFTTestFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<const DFTTestData *>(instanceData);

    vsapi->freeNode(d->node);

    showError(cuCtxPushCurrent(d->context));

    for (const auto & [_, thread_data] : d->thread_data) {
        std::free(thread_data.padded);
        showError(cuMemFreeHost(thread_data.h_spatial));
    }

    delete d;

    showError(cuCtxPopCurrent(nullptr));

    showError(cuDevicePrimaryCtxRelease(d->device));
}

static void VS_CC DFTTestCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    bool context_retained = false;
    bool context_pushed = false;

    context_releaser context_releaser { &context_retained };

    // release before pop context
    context_popper context_popper { &context_pushed };

    auto d = std::make_unique<DFTTestData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(d->node);

    auto user_kernel = vsapi->propGetData(in, "kernel", 0, nullptr);

    int error;

    d->radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &error));
    if (error) {
        d->radius = 0;
    }

    d->block_size = int64ToIntS(vsapi->propGetInt(in, "block_size", 0, &error));
    if (error) {
        d->block_size = 8;
    }

    node_freer node_freer { vsapi, d->node };

    {
        if (vsapi->propNumElements(in, "window") != (2 * d->radius + 1) * square(d->block_size)) {
            vsapi->setError(out, "\"window\" must contain exactly (2*radius+1)*block_size^2 number of elements");
            return ;
        }
        d->window = std::make_unique<float []>((2 * d->radius + 1) * square(d->block_size) * sizeof(float));
        auto array = vsapi->propGetFloatArray(in, "window", nullptr);
        for (int i = 0; i < (2 * d->radius + 1) * square(d->block_size); i++) {
            d->window[i] = static_cast<float>(array[i]);
        }
    }

    d->block_step = int64ToIntS(vsapi->propGetInt(in, "block_step", 0, &error));
    if (error) {
        d->block_step = d->block_size;
    }

    int device_id = int64ToIntS(vsapi->propGetInt(in, "device_id", 0, &error));
    if (error) {
        device_id = 0;
    }

    if (auto result = cuInit(0); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuInit(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuDeviceGet(&d->device, device_id); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDeviceGet(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuDevicePrimaryCtxRetain(&d->context, d->device); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDevicePrimaryCtxRetain(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    context_retained = true;
    context_releaser.device = d->device;

    if (auto result = cuCtxPushCurrent(d->context); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuCtxPushCurrent(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    context_pushed = true;

    auto compilation = compile(user_kernel, d->device);
    if (std::holds_alternative<std::string>(compilation)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] compile(): " << std::get<std::string>(compilation);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    d->module = std::get<CUmodule>(compilation);
    if (auto result = cuModuleGetFunction(&d->filter_kernel, d->module, "frequency_filtering"); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuModuleGetFunction(frequency_filtering): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    int num_sms;
    if (auto result = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, d->device); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDeviceGetAttribute(multiprocessor_count): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    int max_blocks_per_sm;
    if (auto result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, d->filter_kernel, 128, 0); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuOccupancyMaxActiveBlocksPerMultiprocessor(frequency_filtering): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    d->filter_num_blocks = num_sms * max_blocks_per_sm;

    if (auto result = cuStreamCreate(&d->stream.data, CU_STREAM_NON_BLOCKING); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuStreamCreate(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuMemAlloc(&d->d_spatial.data, (2 * d->radius + 1) * calc_pad_num(vi->height, d->block_size, d->block_step) * calc_pad_num(vi->width, d->block_size, d->block_step) * square(d->block_size) * sizeof(float)); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemAlloc(spatial): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuMemAlloc(&d->d_frequency.data, (2 * d->radius + 1) * calc_pad_num(vi->height, d->block_size, d->block_step) * calc_pad_num(vi->width, d->block_size, d->block_step) * d->block_size * (d->block_size / 2 + 1) * sizeof(cufftComplex)); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemAlloc(frequency): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    int batch = calc_pad_num(vi->height, d->block_size, d->block_step) * calc_pad_num(vi->width, d->block_size, d->block_step);
    if (auto result = cufftCreate(&d->rfft_handle.data); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftCreate(rfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    if (d->radius == 0) {
        std::array<int, 2> n { d->block_size, d->block_size };
        if (auto result = cufftPlanMany(&d->rfft_handle.data, 2, n.data(), nullptr, 1, square(d->block_size), nullptr, 1, d->block_size * (d->block_size / 2 + 1), CUFFT_R2C, batch); !success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftPlanMany(rfft2): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
    } else { // radius != 0
        std::array<int, 3> n { 2 * d->radius + 1, d->block_size, d->block_size };
        if (auto result = cufftPlanMany(&d->rfft_handle.data, 3, n.data(), nullptr, 1, (2 * d->radius + 1) * square(d->block_size), nullptr, 1, (2 * d->radius + 1) * d->block_size * (d->block_size / 2 + 1), CUFFT_R2C, batch); !success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftPlanMany(rfft3): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
    }
    if (auto result = cufftSetStream(d->rfft_handle, d->stream); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftSetStream(rfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cufftCreate(&d->irfft_handle.data); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftCreate(irfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    if (d->radius == 0) {
        std::array<int, 2> n { d->block_size, d->block_size };
        auto result = cufftPlanMany(&d->irfft_handle.data, 2, n.data(), nullptr, 1, d->block_size * (d->block_size / 2 + 1), nullptr, 1, square(d->block_size), CUFFT_C2R, batch);
        if (!success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftPlanMany(irfft2): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
    } else { // radius != 0
        std::array<int, 3> n { 2 * d->radius + 1, d->block_size, d->block_size };
        auto result = cufftPlanMany(&d->irfft_handle.data, 3, n.data(), nullptr, 1, (2 * d->radius + 1) * d->block_size * (d->block_size / 2 + 1), nullptr, 1, (2 * d->radius + 1) * square(d->block_size), CUFFT_C2R, batch);
        if (!success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftPlanMany(irfft3): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
    }
    if (auto result = cufftSetStream(d->irfft_handle, d->stream); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftSetStream(irfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    VSCoreInfo info;
    vsapi->getCoreInfo2(core, &info);
    d->num_uninitialized_threads.store(info.numThreads, std::memory_order_relaxed);
    d->thread_data.reserve(info.numThreads);

    vsapi->createFilter(
        in, out, "DFTTest",
        DFTTestInit, DFTTestGetFrame, DFTTestFree,
        fmParallel, 0, d.release(), core
    );

    node_freer.release();
    context_releaser.release();
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("io.github.amusementclub.dfttest2_cuda", "dfttest2_cuda", "DFTTest2 (CUDA)", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc(
        "DFTTest",
        "clip:clip;"
        "kernel:data[];"
        "window:float[];"
        "radius:int:opt;"
        "block_size:int:opt;"
        "block_step:int:opt;"
        "device_id:int:opt;",
        DFTTestCreate, nullptr, plugin
    );
}
