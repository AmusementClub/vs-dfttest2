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

template <typename T>
static void reflection_padding_impl(
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
        std::memcpy(&dst[y * pad_width], &dst[(offset_y * 2 - y) * pad_width], pad_width * sizeof(T));
    }

    // copy bottom region
    for (int y = offset_y + height; y < pad_height; y++) {
        std::memcpy(&dst[y * pad_width], &dst[(2 * (offset_y + height) - 2 - y) * pad_width], pad_width * sizeof(T));
    }
}

static void reflection_padding(
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

static std::variant<CUmodule, std::string> compile(
    const char * user_kernel,
    CUdevice device,
    int radius,
    int block_size,
    int block_step,
    int warp_size,
    int warps_per_block,
    int sample_type,
    int bits_per_sample
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
    kernel_source << "#define RADIUS " << radius << '\n';
    kernel_source << "#define BLOCK_SIZE " << block_size << '\n';
    kernel_source << "#define BLOCK_STEP " << block_step << '\n';
    kernel_source << "#define WARP_SIZE " << warp_size << '\n';
    kernel_source << "#define WARPS_PER_BLOCK " << warps_per_block << '\n';
    if (sample_type == stInteger) {
        int bytes_per_sample = bits_per_sample / 8;
        const char * type;
        if (bytes_per_sample == 1) {
            type = "unsigned char";
        } else if (bytes_per_sample == 2) {
            type = "unsigned short";
        } else if (bytes_per_sample == 4) {
            type = "unsigned int";
        }
        kernel_source << "#define TYPE " << type << '\n';
        kernel_source << "#define SCALE " << 1.0 / (1 << (bits_per_sample - 8)) << '\n';
        kernel_source << "#define PEAK " << ((1 << bits_per_sample) - 1) << '\n';
    } else if (sample_type == stFloat) {
        if (bits_per_sample == 32) {
            kernel_source << "#define TYPE float\n";
        }
        kernel_source << "#define SCALE 255.0\n";
    }
    kernel_source << user_kernel << '\n';
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
    uint8_t * h_padded; // shape: (pad_height, pad_width)
};


struct DFTTestData {
    VSNodeRef * node;
    int radius;
    int block_size;
    int block_step;
    std::array<bool, 3> process;
    CUdevice device; // device_id
    int warp_size;
    int warps_per_block = 4; // most existing devices contain four schedulers per sm

    CUcontext context; // use primary stream for interoperability
    Resource<CUstream, cuStreamDestroyCustom> stream;
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_padded; // shape: (pad_height, pad_width)
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_spatial; // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size)
    Resource<CUdeviceptr, cuMemFreeCustom, true> d_frequency; // shape: (vertical_num, horizontal_num, 2*radius+1, block_size, block_size/2+1)
    std::mutex lock; // TODO: replace by `num_streams`

    // 2-D or 3-D, depends on radius
    Resource<cufftHandle, cufftDestroyCustom, true> rfft_handle;
    Resource<cufftHandle, cufftDestroyCustom, true> irfft_handle;
    Resource<cufftHandle, cufftDestroyCustom, true> subsampled_rfft_handle;
    Resource<cufftHandle, cufftDestroyCustom, true> subsampled_irfft_handle;

    Resource<CUmodule, cuModuleUnloadCustom> module;
    CUfunction filter_kernel;
    int filter_num_blocks;
    CUfunction im2col_kernel;
    int im2col_num_blocks;
    CUfunction col2im_kernel;

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
                    (2 * d->radius + 1) *
                    calc_pad_size(vi->height, d->block_size, d->block_step) *
                    calc_pad_size(vi->width, d->block_size, d->block_step) *
                    vi->format->bytesPerSample
                );

                if (auto result = cuMemHostAlloc(reinterpret_cast<void **>(&thread_data.h_padded), padded_size, 0); result != CUDA_SUCCESS) {
                    std::ostringstream message;
                    message << '[' << __LINE__ << "] cuMemHostAlloc(h_padded) error";
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
            if (!d->process[plane]) {
                continue;
            }

            int width = vsapi->getFrameWidth(src_center_frame.get(), plane);
            int height = vsapi->getFrameHeight(src_center_frame.get(), plane);
            int stride = vsapi->getStride(src_center_frame.get(), plane) / vi->format->bytesPerSample;

            bool subsampled = vi->format->subSamplingW != 0 || vi->format->subSamplingW != 0;
            auto & rfft_handle = (plane == 0 || !subsampled) ? d->rfft_handle : d->subsampled_rfft_handle;
            auto & irfft_handle = (plane == 0 || !subsampled) ? d->irfft_handle : d->subsampled_irfft_handle;

            auto padded_size_spatial = (
                calc_pad_size(height, d->block_size, d->block_step) *
                calc_pad_size(width, d->block_size, d->block_step)
            );

            for (int i = 0; i < 2 * d->radius + 1; i++) {
                auto srcp = vsapi->getReadPtr(src_frames[i].get(), plane);
                reflection_padding(
                    &thread_data.h_padded[(i * padded_size_spatial) * vi->format->bytesPerSample],
                    srcp,
                    width, height, stride,
                    d->block_size, d->block_step,
                    vi->format->bytesPerSample
                );
            }

            {
                std::lock_guard lock { d->lock };

                if (auto result = cuMemcpyHtoDAsync(d->d_padded.data, thread_data.h_padded, (2 * d->radius + 1) * padded_size_spatial * vi->format->bytesPerSample, d->stream); !success(result)) {
                    std::ostringstream message;
                    const char * error_message;
                    showError(cuGetErrorString(result, &error_message));
                    message << '[' << __LINE__ << "] cuMemcpyHtoDAsync(): " << error_message;
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
                {
                    void * params[] { &d->d_spatial.data, &d->d_padded.data, &width, &height };
                    if (auto result = cuLaunchKernel(d->im2col_kernel, static_cast<unsigned int>(d->im2col_num_blocks), 1, 1, d->warps_per_block * d->warp_size, 1, 1, 0, d->stream, params, nullptr); !success(result)) {
                        std::ostringstream message;
                        const char * error_message;
                        showError(cuGetErrorString(result, &error_message));
                        message << '[' << __LINE__ << "] cuLaunchKernel(im2col): " << error_message;
                        vsapi->setFilterError(message.str().c_str(), frameCtx);
                        return nullptr;
                    }
                }
                if (auto result = cufftExecR2C(rfft_handle, reinterpret_cast<cufftReal *>(d->d_spatial.data), reinterpret_cast<cufftComplex *>(d->d_frequency.data)); !success(result)) {
                    std::ostringstream message;
                    message << '[' << __LINE__ << "] cufft(rfft): " << cufftGetErrorString(result);
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
                {
                    int num_blocks = calc_pad_num(height, d->block_size, d->block_step) * calc_pad_num(width, d->block_size, d->block_step);
                    void * params[] { &d->d_frequency.data, &num_blocks };
                    if (auto result = cuLaunchKernel(d->filter_kernel, static_cast<unsigned int>(d->filter_num_blocks), 1, 1, d->warps_per_block * d->warp_size, 1, 1, 0, d->stream, params, nullptr); !success(result)) {
                        std::ostringstream message;
                        const char * error_message;
                        showError(cuGetErrorString(result, &error_message));
                        message << '[' << __LINE__ << "] cuLaunchKernel(frequency_filtering): " << error_message;
                        vsapi->setFilterError(message.str().c_str(), frameCtx);
                        return nullptr;
                    }
                }
                if (auto result = cufftExecC2R(irfft_handle, reinterpret_cast<cufftComplex *>(d->d_frequency.data), reinterpret_cast<cufftReal *>(d->d_spatial.data)); !success(result)) {
                    std::ostringstream message;
                    message << '[' << __LINE__ << "] cufft(irfft): " << cufftGetErrorString(result);
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
                {
                    void * params[] { &d->d_padded.data, &d->d_spatial.data, &width, &height };
                    unsigned int vertical_size = calc_pad_size(height, d->block_size, d->block_step);
                    unsigned int horizontal_size = calc_pad_size(width, d->block_size, d->block_step);
                    if (auto result = cuLaunchKernel(d->col2im_kernel, (horizontal_size + d->warp_size - 1) / d->warp_size, (vertical_size + d->warps_per_block - 1) / d->warps_per_block, 1, d->warp_size, d->warps_per_block, 1, 0, d->stream, params, nullptr); !success(result)) {
                        std::ostringstream message;
                        const char * error_message;
                        showError(cuGetErrorString(result, &error_message));
                        message << '[' << __LINE__ << "] cuLaunchKernel(col2im): " << error_message;
                        vsapi->setFilterError(message.str().c_str(), frameCtx);
                        return nullptr;
                    }
                }
                {
                    const CUDA_MEMCPY3D config {
                        .srcXInBytes = static_cast<size_t>((calc_pad_size(width, d->block_size, d->block_step) - width) / 2 * vi->format->bytesPerSample),
                        .srcY = static_cast<size_t>((calc_pad_size(height, d->block_size, d->block_step) - height) / 2),
                        .srcZ = static_cast<size_t>(d->radius),
                        .srcMemoryType = CU_MEMORYTYPE_DEVICE,
                        .srcDevice = d->d_padded.data,
                        .srcPitch = static_cast<size_t>(calc_pad_size(width, d->block_size, d->block_step) * vi->format->bytesPerSample),
                        .srcHeight = static_cast<size_t>(calc_pad_size(height, d->block_size, d->block_step)),
                        .dstXInBytes = static_cast<size_t>((calc_pad_size(width, d->block_size, d->block_step) - width) / 2 * vi->format->bytesPerSample),
                        .dstY = static_cast<size_t>((calc_pad_size(height, d->block_size, d->block_step) - height) / 2),
                        .dstZ = 0, // vs_bitblt(dstp) copies from the 0-th slice
                        .dstMemoryType = CU_MEMORYTYPE_HOST,
                        .dstHost = thread_data.h_padded,
                        .dstPitch = static_cast<size_t>(calc_pad_size(width, d->block_size, d->block_step) * vi->format->bytesPerSample),
                        .dstHeight = static_cast<size_t>(calc_pad_size(height, d->block_size, d->block_step)),
                        .WidthInBytes = static_cast<size_t>(width * vi->format->bytesPerSample),
                        .Height = static_cast<size_t>(height),
                        .Depth = 1
                    };
                    if (auto result = cuMemcpy3DAsync(&config, d->stream); !success(result)) {
                        std::ostringstream message;
                        const char * error_message;
                        showError(cuGetErrorString(result, &error_message));
                        message << '[' << __LINE__ << "] cuMemcpy3DAsync(DtoH): " << error_message;
                        vsapi->setFilterError(message.str().c_str(), frameCtx);
                        return nullptr;
                    }
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

            int pad_width = calc_pad_size(width, d->block_size, d->block_step);
            int pad_height = calc_pad_size(height, d->block_size, d->block_step);
            int offset_y = (pad_height - height) / 2;
            int offset_x = (pad_width - width) / 2;

            auto dstp = vsapi->getWritePtr(dst_frame.get(), plane);
            vs_bitblt(
                dstp, stride * vi->format->bytesPerSample,
                &thread_data.h_padded[(offset_y * pad_width + offset_x) * vi->format->bytesPerSample], pad_width * vi->format->bytesPerSample,
                width * vi->format->bytesPerSample, height
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
        showError(cuMemFreeHost(thread_data.h_padded));
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

    d->block_step = int64ToIntS(vsapi->propGetInt(in, "block_step", 0, &error));
    if (error) {
        d->block_step = d->block_size;
    }

    int num_planes_args = vsapi->propNumElements(in, "planes");
    d->process.fill(num_planes_args <= 0);
    for (int i = 0; i < num_planes_args; ++i) {
        int plane = static_cast<int>(vsapi->propGetInt(in, "planes", i, nullptr));

        if (plane < 0 || plane >= vi->format->numPlanes) {
            vsapi->setError(out, "plane index out of range");
            return ;
        }

        if (d->process[plane]) {
            vsapi->setError(out, "plane specified twice");
            return ;
        }
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

    if (auto result = cuDeviceGetAttribute(&d->warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, d->device); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDeviceGetAttribute(warp_size): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    auto compilation = compile(user_kernel, d->device, d->radius, d->block_size, d->block_step, d->warp_size, d->warps_per_block, vi->format->sampleType, vi->format->bitsPerSample);
    if (std::holds_alternative<std::string>(compilation)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] compile(): " << std::get<std::string>(compilation);
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

    d->module = std::get<CUmodule>(compilation);

    if (auto result = cuModuleGetFunction(&d->filter_kernel, d->module, "frequency_filtering"); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuModuleGetFunction(frequency_filtering): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    } else {
        int max_blocks_per_sm;
        if (auto result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, d->filter_kernel, d->warps_per_block * d->warp_size, 0); !success(result)) {
            std::ostringstream message;
            const char * error_message;
            showError(cuGetErrorString(result, &error_message));
            message << '[' << __LINE__ << "] cuOccupancyMaxActiveBlocksPerMultiprocessor(frequency_filtering): " << error_message;
            vsapi->setError(out, message.str().c_str());
            return ;
        }
        d->filter_num_blocks = num_sms * max_blocks_per_sm;
    }

    if (auto result = cuModuleGetFunction(&d->im2col_kernel, d->module, "im2col"); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuModuleGetFunction(im2col): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    } else {
        int max_blocks_per_sm;
        if (auto result = cuOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, d->im2col_kernel, d->warps_per_block * d->warp_size, 0); !success(result)) {
            std::ostringstream message;
            const char * error_message;
            showError(cuGetErrorString(result, &error_message));
            message << '[' << __LINE__ << "] cuOccupancyMaxActiveBlocksPerMultiprocessor(im2col): " << error_message;
            vsapi->setError(out, message.str().c_str());
            return ;
        }
        d->im2col_num_blocks = num_sms * max_blocks_per_sm;
    }

    if (auto result = cuModuleGetFunction(&d->col2im_kernel, d->module, "col2im"); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuModuleGetFunction(col2im): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuStreamCreate(&d->stream.data, CU_STREAM_NON_BLOCKING); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuStreamCreate(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuMemAlloc(&d->d_padded.data, (2 * d->radius + 1) * calc_pad_size(vi->height, d->block_size, d->block_step) * calc_pad_size(vi->width, d->block_size, d->block_step) * vi->format->bytesPerSample); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemAlloc(padded): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuMemAlloc(&d->d_spatial.data, calc_pad_num(vi->height, d->block_size, d->block_step) * calc_pad_num(vi->width, d->block_size, d->block_step) * (2 * d->radius + 1) * square(d->block_size) * sizeof(cufftReal)); !success(result)) {
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

    if (vi->format->subSamplingW != 0 || vi->format->subSamplingH != 0) {
        int subsampled_batch = calc_pad_num(vi->height >> vi->format->subSamplingH, d->block_size, d->block_step) * calc_pad_num(vi->width >> vi->format->subSamplingW, d->block_size, d->block_step);

        if (auto result = cufftCreate(&d->subsampled_rfft_handle.data); !success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftCreate(subsampled_rfft): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
        if (d->radius == 0) {
            std::array<int, 2> n { d->block_size, d->block_size };
            if (auto result = cufftPlanMany(&d->subsampled_rfft_handle.data, 2, n.data(), nullptr, 1, square(d->block_size), nullptr, 1, d->block_size * (d->block_size / 2 + 1), CUFFT_R2C, subsampled_batch); !success(result)) {
                std::ostringstream message;
                message << '[' << __LINE__ << "] cufftPlanMany(subsampled_rfft2): " << cufftGetErrorString(result);
                vsapi->setError(out, message.str().c_str());
                return ;
            }
        } else { // radius != 0
            std::array<int, 3> n { 2 * d->radius + 1, d->block_size, d->block_size };
            if (auto result = cufftPlanMany(&d->subsampled_rfft_handle.data, 3, n.data(), nullptr, 1, (2 * d->radius + 1) * square(d->block_size), nullptr, 1, (2 * d->radius + 1) * d->block_size * (d->block_size / 2 + 1), CUFFT_R2C, subsampled_batch); !success(result)) {
                std::ostringstream message;
                message << '[' << __LINE__ << "] cufftPlanMany(subsampled_rfft3): " << cufftGetErrorString(result);
                vsapi->setError(out, message.str().c_str());
                return ;
            }
        }
        if (auto result = cufftSetStream(d->subsampled_rfft_handle, d->stream); !success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftSetStream(subsampled_rfft): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }

        if (auto result = cufftCreate(&d->subsampled_irfft_handle.data); !success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftCreate(subsampled_irfft): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
        if (d->radius == 0) {
            std::array<int, 2> n { d->block_size, d->block_size };
            auto result = cufftPlanMany(&d->subsampled_irfft_handle.data, 2, n.data(), nullptr, 1, d->block_size * (d->block_size / 2 + 1), nullptr, 1, square(d->block_size), CUFFT_C2R, subsampled_batch);
            if (!success(result)) {
                std::ostringstream message;
                message << '[' << __LINE__ << "] cufftPlanMany(subsampled_irfft2): " << cufftGetErrorString(result);
                vsapi->setError(out, message.str().c_str());
                return ;
            }
        } else { // radius != 0
            std::array<int, 3> n { 2 * d->radius + 1, d->block_size, d->block_size };
            auto result = cufftPlanMany(&d->subsampled_irfft_handle.data, 3, n.data(), nullptr, 1, (2 * d->radius + 1) * d->block_size * (d->block_size / 2 + 1), nullptr, 1, (2 * d->radius + 1) * square(d->block_size), CUFFT_C2R, subsampled_batch);
            if (!success(result)) {
                std::ostringstream message;
                message << '[' << __LINE__ << "] cufftPlanMany(subsampled_irfft3): " << cufftGetErrorString(result);
                vsapi->setError(out, message.str().c_str());
                return ;
            }
        }
        if (auto result = cufftSetStream(d->subsampled_irfft_handle, d->stream); !success(result)) {
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftSetStream(subsampled_irfft): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
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

static void VS_CC RDFT(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    bool context_retained = false;
    bool context_pushed = false;

    context_releaser context_releaser { &context_retained };

    // release before pop context
    context_popper context_popper { &context_pushed };

    auto set_error = [vsapi, out](const char * error_message) -> void {
        vsapi->setError(out, error_message);
    };

    int ndim = vsapi->propNumElements(in, "shape");
    if (ndim != 1 && ndim != 2 && ndim != 3) {
        return set_error("\"shape\" must be an array of ints with 1, 2 or 3 values");
    }

    std::array<int, 3> shape;
    {
        auto shape_array = vsapi->propGetIntArray(in, "shape", nullptr);
        for (int i = 0; i < ndim; i++) {
            shape[i] = int64ToIntS(shape_array[i]);
        }
    }

    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    int complex_size = shape[ndim - 1] / 2 + 1;
    for (int i = 0; i < ndim - 1; i++) {
        complex_size *= shape[i];
    }

    if (vsapi->propNumElements(in, "data") != size) {
        return set_error("cannot reshape array");
    }

    auto data = vsapi->propGetFloatArray(in, "data", nullptr);

    int error;
    int device_id = static_cast<int>(vsapi->propGetInt(in, "device_id", 0, &error));
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

    CUdevice device;

    if (auto result = cuDeviceGet(&device, device_id); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDeviceGet(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    CUcontext context;

    if (auto result = cuDevicePrimaryCtxRetain(&context, device); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDevicePrimaryCtxRetain(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    context_retained = true;
    context_releaser.device = device;

    if (auto result = cuCtxPushCurrent(context); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuCtxPushCurrent(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    context_pushed = true;

    Resource<CUstream, cuStreamDestroyCustom> stream {};
    if (auto result = cuStreamCreate(&stream.data, CU_STREAM_NON_BLOCKING); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuStreamCreate(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    Resource<CUdeviceptr, cuMemFreeCustom, true> d_spatial {};
    if (auto result = cuMemAlloc(&d_spatial.data, size * sizeof(cufftDoubleReal)); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemAlloc(spatial): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    Resource<CUdeviceptr, cuMemFreeCustom, true> d_frequency {};
    if (auto result = cuMemAlloc(&d_frequency.data, complex_size * sizeof(cufftDoubleComplex)); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemAlloc(frequency): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    Resource<double *, cuMemFreeHostCustom> h_frequency {};
    if (auto result = cuMemHostAlloc(reinterpret_cast<void **>(&h_frequency.data), complex_size * sizeof(cufftDoubleComplex), 0); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemHostAlloc(frequency): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    Resource<cufftHandle, cufftDestroyCustom, true> rfft_handle;
    if (auto result = cufftCreate(&rfft_handle.data); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftCreate(rfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    if (auto result = cufftPlanMany(&rfft_handle.data, ndim, shape.data(), nullptr, 1, 0, nullptr, 1, 0, CUFFT_D2Z, 1); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftPlanMany(rfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    if (auto result = cufftSetStream(rfft_handle, stream); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftSetStream(rfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuMemcpyHtoDAsync(d_spatial, data, size * sizeof(cufftDoubleReal), stream); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemcpyHtoDAsync(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    if (auto result = cufftExecD2Z(rfft_handle, reinterpret_cast<cufftDoubleReal *>(d_spatial.data), reinterpret_cast<cufftDoubleComplex *>(d_frequency.data)); !success(result)) {
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufft(rfft): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    if (auto result = cuMemcpyDtoHAsync(h_frequency, d_frequency, complex_size * sizeof(cufftDoubleComplex), stream); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemcpyDtoHAsync(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    if (auto result = cuStreamSynchronize(stream); !success(result)) {
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuStreamSynchronize(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    vsapi->propSetFloatArray(out, "ret", h_frequency, complex_size * 2);
}

static void VS_CC ToSingle(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto data = vsapi->propGetFloatArray(in, "data", nullptr);
    int num = vsapi->propNumElements(in, "data");

    auto converted_data = std::make_unique<double []>(num);
    for (int i = 0; i < num; i++) {
        converted_data[i] = static_cast<float>(data[i]);
    }

    if (num == 1) {
        vsapi->propSetFloat(out, "ret", converted_data[0], paReplace);
    } else {
        vsapi->propSetFloatArray(out, "ret", converted_data.get(), num);
    }
}

VS_EXTERNAL_API(void)
VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc("io.github.amusementclub.dfttest2_cuda", "dfttest2_cuda", "DFTTest2 (CUDA)", VAPOURSYNTH_API_VERSION, 1, plugin);

    registerFunc(
        "DFTTest",
        "clip:clip;"
        "kernel:data[];"
        "radius:int:opt;"
        "block_size:int:opt;"
        "block_step:int:opt;"
        "planes:int[]:opt;"
        "device_id:int:opt;",
        DFTTestCreate, nullptr, plugin
    );

    registerFunc(
        "RDFT",
        "data:float[];"
        "shape:int[];"
        "device_id:int:opt;",
        RDFT, nullptr, plugin
    );

    registerFunc(
        "ToSingle",
        "data:float[];",
        ToSingle, nullptr, plugin
    );
}
