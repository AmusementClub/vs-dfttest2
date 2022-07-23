#include <array>
#include <atomic>
#include <algorithm>
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

template <typename T>
static T square(const T & x) {
    return x * x;
}

static int calc_size(int width, int block_size, int block_step) {
    return (width + block_size - 1) / block_step;
}

static void im2col(
    float * VS_RESTRICT dst, // shape: (vertical_num, horizontal_num, block_size, block_size)
    const float ** VS_RESTRICT srcs, // shape: (radius, height, stride)
    int radius,
    int width, int height, int stride,
    int block_size, int block_step
) {

    int horizontal_num = calc_size(width, block_size, block_step);
    int vertical_num = calc_size(height, block_size, block_step);

    for (int i = 0; i < vertical_num; i++) {
        int v_index = std::min(i * block_step, height - block_size) * stride;
        for (int j = 0; j < horizontal_num; j++) {
            int h_index = std::min(j * block_step, width - block_size);
            auto src = *srcs + v_index + h_index;
            vs_bitblt(
                dst, block_size * sizeof(float), 
                src, stride * sizeof(float), 
                block_size * sizeof(float), block_size
            );
            dst += square(block_size);
        }
    }
}

static void init_multiplier(
    float * multiplier, // shape: (height, width)
    int width, int height,
    int block_size, int block_step
) {

    std::memset(multiplier, 0, height * width * sizeof(float));

    int horizontal_num = calc_size(width, block_size, block_step);
    int vertical_num = calc_size(height, block_size, block_step);
    for (int i = 0; i < vertical_num; i++) {
        int v_index = std::min(i * block_step, height - block_size) * width;
        for (int j = 0; j < horizontal_num; j++) {
            int h_index = std::min(j * block_step, width - block_size);
            for (int k = 0; k < block_size; k++) {
                for (int l = 0; l < block_size; l++) {
                    multiplier[(v_index + k * width) + (h_index + l)] += 1.0f;
                }
            }
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // cuFFT performs un-normalized FFTs
            multiplier[i * width + j] = 1.0f / (multiplier[i * width + j] * square(block_size));
        }
    }
}

static void col2im(
    float * VS_RESTRICT dst, // shape: (height, stride)
    const float * VS_RESTRICT src, // shape: (vertical_num, horizontal_num, block_size, block_size)
    const float * VS_RESTRICT multiplier, // shape: (height, width)
    int width, int height, int stride,
    int block_size, int block_step
) {

    std::memset(dst, 0, height * stride * sizeof(float));

    int horizontal_num = calc_size(width, block_size, block_step);
    int vertical_num = calc_size(height, block_size, block_step);
    for (int i = 0; i < vertical_num; i++) {
        int v_index_src = i * horizontal_num * square(block_size);
        int v_index_dst = std::min(i * block_step, height - block_size) * stride;
        for (int j = 0; j < horizontal_num; j++) {
            int h_index_src = j * square(block_size);
            int h_index_dst = std::min(j * block_step, width - block_size);
            for (int k = 0; k < block_size; k++) {
                for (int l = 0; l < block_size; l++) {
                    dst[(v_index_dst + k * stride) + (h_index_dst + l)] += src[v_index_src + h_index_src + k * block_size + l];
                }
            }
        }
    }
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            dst[i * stride + j] *= multiplier[i * width + j];
        }
    }
}

static int calc_spatial_size(int width, int height, int block_size, int block_step) {
    return (
        calc_size(height, block_size, block_step) *
        calc_size(width, block_size, block_step) *
        square(block_size) *
        static_cast<int>(sizeof(cufftReal))
    );
}

static int calc_frequency_size(int width, int height, int block_size, int block_step) {
    return (
        calc_size(height, block_size, block_step) *
        calc_size(width, block_size, block_step) *
        block_size * 
        (block_size / 2 + 1) *
        static_cast<int>(sizeof(cufftComplex))
    );
}

static std::variant<CUmodule, std::string> compile(
    const char * kernel,
    // int width, int height, int stride,
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
    kernel_source << kernel_header_template;
    kernel_source << kernel;

    nvrtcProgram program;
    if (auto result = nvrtcCreateProgram(&program, kernel_source.str().c_str(), nullptr, 0, nullptr, nullptr); result != NVRTC_SUCCESS) {
        return std::string{nvrtcGetErrorString(result)};
    }

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

    if (nvrtcCompileProgram(program, static_cast<int>(std::extent_v<decltype(opts)>), opts) != NVRTC_SUCCESS) {
        size_t log_size;
        showError(nvrtcGetProgramLogSize(program, &log_size));
        std::string error_message;
        error_message.resize(log_size);
        showError(nvrtcGetProgramLog(program, error_message.data()));
        showError(nvrtcDestroyProgram(&program));
        return error_message;
    }

    size_t cubin_size;
    if (auto result = nvrtcGetCUBINSize(program, &cubin_size); !success(result)) {
        showError(nvrtcDestroyProgram(&program));
        return std::string{nvrtcGetErrorString(result)};
    }
    auto image = std::make_unique<char[]>(cubin_size);
    if (auto result = nvrtcGetCUBIN(program, image.get()); !success(result)) {
        showError(nvrtcDestroyProgram(&program));
        return std::string{nvrtcGetErrorString(result)};
    }

    showError(nvrtcDestroyProgram(&program));

    CUmodule module;
    if (auto result = cuModuleLoadData(&module, image.get()); !success(result)) {
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        return std::string{error_message};
    }

    return module;
}


struct DFTTestData {
    VSNodeRef * node;
    int block_size;
    int block_step;
    CUdevice device; // device_id

    CUcontext context; // use primary stream for interoperability
    CUstream stream;
    CUdeviceptr d_spatial; // (vertical_num, horizontal_num, block_size, block_size)
    CUdeviceptr d_frequency; // (vertical_num, horizontal_num, block_size, block_size/2+1)

    cufftHandle rfft2d_handle;
    cufftHandle irfft2d_handle;

    CUmodule module;
    CUfunction kernel;

    std::atomic<int> num_uninitialized_threads;
    std::unordered_map<std::thread::id, float *> h_spatial;
    std::shared_mutex h_spatial_lock;
    std::unique_ptr<float []> multiplier;
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
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        if (auto result = cuCtxPushCurrent(d->context); !success(result)) {
            std::ostringstream message;
            const char * error_message;
            showError(cuGetErrorString(result, &error_message));
            message << '[' << __LINE__ << "] cuCtxPushCurrent(): " << error_message;
            vsapi->setFilterError(message.str().c_str(), frameCtx);
            return nullptr;
        }

        auto vi = vsapi->getVideoInfo(d->node);

        float * h_spatial;

        auto thread_id = std::this_thread::get_id();
        if (d->num_uninitialized_threads.load(std::memory_order_acquire) > 0) {
            bool initialized = true;

            d->h_spatial_lock.lock_shared();
            try {
                h_spatial = d->h_spatial.at(thread_id);
            } catch (const std::out_of_range &) {
                initialized = false;
            }
            d->h_spatial_lock.unlock_shared();

            if (!initialized) {
                auto size = calc_size(vi->height, d->block_size, d->block_step) *
                    calc_size(vi->width, d->block_size, d->block_step) *
                    square(d->block_size) *
                    sizeof(float);

                if (auto result = cuMemHostAlloc(reinterpret_cast<void **>(&h_spatial), size, 0); !success(result)) {
                    std::ostringstream message;
                    const char * error_message;
                    showError(cuGetErrorString(result, &error_message));
                    message << '[' << __LINE__ << "] cuMemHostAlloc(): " << error_message;
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }

                {
                    std::lock_guard _ { d->h_spatial_lock };
                    d->h_spatial.emplace(thread_id, h_spatial);
                }
                d->num_uninitialized_threads.fetch_sub(1, std::memory_order_release);
            }
        } else {
            const auto & const_buffer = d->h_spatial;
            h_spatial = const_buffer.at(thread_id);
        }
        auto src_frame = vsapi->getFrameFilter(n, d->node, frameCtx);
        auto format = vsapi->getFrameFormat(src_frame);

        auto dst_frame = vsapi->newVideoFrame(format, vi->width, vi->height, src_frame, core);

        for (int plane = 0; plane < format->numPlanes; plane++) {
            int width = vsapi->getFrameWidth(src_frame, plane);
            int height = vsapi->getFrameHeight(src_frame, plane);
            int stride = vsapi->getStride(src_frame, plane) / sizeof(float);

            auto srcp = vsapi->getReadPtr(src_frame, plane);
            im2col(
                h_spatial, 
                reinterpret_cast<const float **>(&srcp), 
                0, 
                width, height, stride, 
                d->block_size, d->block_step
            );

            auto spatial_size = calc_spatial_size(width, height, d->block_size, d->block_step);
            if (auto result = cuMemcpyHtoDAsync(d->d_spatial,h_spatial, spatial_size, d->stream); !success(result)) {
                vsapi->freeFrame(dst_frame);
                vsapi->freeFrame(src_frame);
                std::ostringstream message;
                const char * error_message;
                showError(cuGetErrorString(result, &error_message));
                message << '[' << __LINE__ << "] cuMemcpyHtoDAsync(): " << error_message;
                vsapi->setFilterError(message.str().c_str(), frameCtx);
                return nullptr;
            }
            if (auto result = cufftExecR2C(d->rfft2d_handle, reinterpret_cast<cufftReal *>(d->d_spatial), reinterpret_cast<cufftComplex *>(d->d_frequency)); !success(result)) {
                vsapi->freeFrame(dst_frame);
                vsapi->freeFrame(src_frame);
                std::ostringstream message;
                message << '[' << __LINE__ << "] cufft(rfft2): " << cufftGetErrorString(result);
                vsapi->setFilterError(message.str().c_str(), frameCtx);
                return nullptr;
            }
            {
                int n = static_cast<int>(calc_frequency_size(width, height, d->block_size, d->block_step) / sizeof(cufftComplex));
                void * params[] { &d->d_frequency, &n, &d->block_size };
                if (auto result = cuLaunchKernel(d->kernel, static_cast<unsigned int>((n + 127) / 128), 1, 1, 128, 1, 1, 0, d->stream, params, nullptr); !success(result)) {
                    vsapi->freeFrame(dst_frame);
                    vsapi->freeFrame(src_frame);
                    std::ostringstream message;
                    const char * error_message;
                    showError(cuGetErrorString(result, &error_message));
                    message << '[' << __LINE__ << "] cuLaunchKernel(): " << error_message;
                    vsapi->setFilterError(message.str().c_str(), frameCtx);
                    return nullptr;
                }
            }
            if (auto result = cufftExecC2R(d->irfft2d_handle, reinterpret_cast<cufftComplex *>(d->d_frequency), reinterpret_cast<cufftReal *>(d->d_spatial)); !success(result)) {
                vsapi->freeFrame(dst_frame);
                vsapi->freeFrame(src_frame);
                std::ostringstream message;
                message << '[' << __LINE__ << "] cufft(irfft2): " << cufftGetErrorString(result);
                vsapi->setFilterError(message.str().c_str(), frameCtx);
                return nullptr;
            }
            if (auto result = cuMemcpyDtoHAsync(h_spatial, d->d_spatial, spatial_size, d->stream); !success(result)) {
                vsapi->freeFrame(dst_frame);
                vsapi->freeFrame(src_frame);
                std::ostringstream message;
                const char * error_message;
                showError(cuGetErrorString(result, &error_message));
                message << '[' << __LINE__ << "] cuMemcpyDtoHAsync(): " << error_message;
                vsapi->setFilterError(message.str().c_str(), frameCtx);
                return nullptr;
            }
            if (auto result = cuStreamSynchronize(d->stream); !success(result)) {
                vsapi->freeFrame(dst_frame);
                vsapi->freeFrame(src_frame);
                std::ostringstream message;
                const char * error_message;
                showError(cuGetErrorString(result, &error_message));
                message << '[' << __LINE__ << "] cuStreamSynchronize(): " << error_message;
                vsapi->setFilterError(message.str().c_str(), frameCtx);
                return nullptr;
            }

            auto dstp = vsapi->getWritePtr(dst_frame, plane);
            col2im(
                reinterpret_cast<float *>(dstp), 
                h_spatial, 
                d->multiplier.get(),
                width, height, stride, 
                d->block_size, d->block_step
            );
        }

        showError(cuCtxPopCurrent(nullptr));

        vsapi->freeFrame(src_frame);

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC DFTTestFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<const DFTTestData *>(instanceData);

    vsapi->freeNode(d->node);

    for (const auto & [_, h_spatial] : d->h_spatial) {
        showError(cuMemFreeHost(h_spatial));
    }

    showError(cufftDestroy(d->irfft2d_handle));
    showError(cufftDestroy(d->rfft2d_handle));
    showError(cuMemFree(d->d_frequency));
    showError(cuMemFree(d->d_spatial));
    showError(cuStreamDestroy(d->stream));
    showError(cuModuleUnload(d->module));
    showError(cuDevicePrimaryCtxRelease(d->device));
}

static void VS_CC DFTTestCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = std::make_unique<DFTTestData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(d->node);

    int error;

    d->block_size = int64ToIntS(vsapi->propGetInt(in, "block_size", 0, &error));
    if (error) {
        d->block_size = 8;
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
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuInit(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuDeviceGet(&d->device, device_id); !success(result)) {
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDeviceGet(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuDevicePrimaryCtxRetain(&d->context, d->device); !success(result)) {
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuDevicePrimaryCtxRetain(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuCtxPushCurrent(d->context); !success(result)) {
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuCtxPushCurrent(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    auto kernel_source = vsapi->propGetData(in, "kernel", 0, nullptr);
    auto compilation = compile(kernel_source, d->device);
    if (std::holds_alternative<std::string>(compilation)) {
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        message << '[' << __LINE__ << "] compile(): " << std::get<std::string>(compilation);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    d->module = std::get<CUmodule>(compilation);
    if (auto result = cuModuleGetFunction(&d->kernel, d->module, "frequency_filtering"); !success(result)) {
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuModuleGetFunction(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuStreamCreate(&d->stream, CU_STREAM_NON_BLOCKING); !success(result)) {
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuStreamCreate(): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuMemAlloc(&d->d_spatial, calc_spatial_size(vi->width, vi->height, d->block_size, d->block_step)); !success(result)) {
        showError(cuStreamDestroy(d->stream));
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemAlloc(spatial): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cuMemAlloc(&d->d_frequency, calc_frequency_size(vi->width, vi->height, d->block_size, d->block_step)); !success(result)) {
        showError(cuMemFree(d->d_spatial));
        showError(cuStreamDestroy(d->stream));
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        const char * error_message;
        showError(cuGetErrorString(result, &error_message));
        message << '[' << __LINE__ << "] cuMemAlloc(frequency): " << error_message;
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    int batch = calc_size(vi->height, d->block_size, d->block_step) * calc_size(vi->width, d->block_size, d->block_step);
    if (auto result = cufftCreate(&d->rfft2d_handle); !success(result)) {
        showError(cuMemFree(d->d_frequency));
        showError(cuMemFree(d->d_spatial));
        showError(cuStreamDestroy(d->stream));
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftCreate(rfft2): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    {
        std::array<int, 2> n { d->block_size, d->block_size };
        auto result = cufftPlanMany(&d->rfft2d_handle, 2, n.data(), nullptr, 1, square(d->block_size), nullptr, 1, d->block_size * (d->block_size / 2 + 1), CUFFT_R2C, batch);
        if (!success(result)) {
            showError(cufftDestroy(d->rfft2d_handle));
            showError(cuMemFree(d->d_frequency));
            showError(cuMemFree(d->d_spatial));
            showError(cuStreamDestroy(d->stream));
            showError(cuModuleUnload(d->module));
            showError(cuCtxPopCurrent(nullptr));
            vsapi->freeNode(d->node);
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftPlanMany(rfft2): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
    }
    if (auto result = cufftSetStream(d->rfft2d_handle, d->stream); !success(result)) {
        showError(cufftDestroy(d->rfft2d_handle));
        showError(cuMemFree(d->d_frequency));
        showError(cuMemFree(d->d_spatial));
        showError(cuStreamDestroy(d->stream));
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftSetStream(rfft2): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    if (auto result = cufftCreate(&d->irfft2d_handle); !success(result)) {
        showError(cufftDestroy(d->rfft2d_handle));
        showError(cuMemFree(d->d_frequency));
        showError(cuMemFree(d->d_spatial));
        showError(cuStreamDestroy(d->stream));
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftCreate(irfft2): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }
    {
        std::array<int, 2> n { d->block_size, d->block_size };
        auto result = cufftPlanMany(&d->irfft2d_handle, 2, n.data(), nullptr, 1, d->block_size * (d->block_size / 2 + 1), nullptr, 1, square(d->block_size), CUFFT_C2R, batch);
        if (!success(result)) {
            showError(cufftDestroy(d->irfft2d_handle));
            showError(cufftDestroy(d->rfft2d_handle));
            showError(cuMemFree(d->d_frequency));
            showError(cuMemFree(d->d_spatial));
            showError(cuStreamDestroy(d->stream));
            showError(cuModuleUnload(d->module));
            showError(cuCtxPopCurrent(nullptr));
            vsapi->freeNode(d->node);
            std::ostringstream message;
            message << '[' << __LINE__ << "] cufftPlanMany(irfft2): " << cufftGetErrorString(result);
            vsapi->setError(out, message.str().c_str());
            return ;
        }
    }
    if (auto result = cufftSetStream(d->irfft2d_handle, d->stream); !success(result)) {
        showError(cufftDestroy(d->irfft2d_handle));
        showError(cufftDestroy(d->rfft2d_handle));
        showError(cuMemFree(d->d_frequency));
        showError(cuMemFree(d->d_spatial));
        showError(cuStreamDestroy(d->stream));
        showError(cuModuleUnload(d->module));
        showError(cuCtxPopCurrent(nullptr));
        vsapi->freeNode(d->node);
        std::ostringstream message;
        message << '[' << __LINE__ << "] cufftSetStream(irfft2): " << cufftGetErrorString(result);
        vsapi->setError(out, message.str().c_str());
        return ;
    }

    VSCoreInfo info;
    vsapi->getCoreInfo2(core, &info);
    d->num_uninitialized_threads.store(info.numThreads, std::memory_order_relaxed);
    d->h_spatial.reserve(info.numThreads);
    d->multiplier.reset(static_cast<float *>(std::malloc(vi->height * vi->width * sizeof(float))));
    init_multiplier(d->multiplier.get(), vi->width, vi->height, d->block_size, d->block_step);

    vsapi->createFilter(
        in, out, "DFTTest", 
        DFTTestInit, DFTTestGetFrame, DFTTestFree, 
        fmParallelRequests, 0, d.release(), core
    );
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
        "device_id:int:opt;", 
        DFTTestCreate, nullptr, plugin
    );
}
