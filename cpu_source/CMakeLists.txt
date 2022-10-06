set(VCL_HOME "${CMAKE_CURRENT_SOURCE_DIR}/vectorclass" CACHE PATH "Path to vector class v2 headers")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    set(CPU_DISPATCH_TARGETS "sse2;avx2;avx512f" CACHE STRING "Dispatch targets")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(CPU_DISPATCH_TARGETS "AVX;AVX2;AVX512" CACHE STRING "Dispatch targets")
endif()

message(STATUS "cpu targets: ${CPU_DISPATCH_TARGETS}")

add_library(dfttest2_cpu MODULE source.cpp ${CMAKE_CURRENT_SOURCE_DIR}/vectorclass/instrset_detect.cpp)

set_target_properties(dfttest2_cpu PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(dfttest2_cpu PRIVATE ${VCL_HOME})

if(PKG_CONFIG_FOUND AND VS_FOUND)
    target_include_directories(dfttest2_cpu PRIVATE ${VS_INCLUDE_DIRS})
    install(TARGETS dfttest2_cpu LIBRARY DESTINATION ${install_dir})
else()
    target_include_directories(dfttest2_cpu PRIVATE ${VS_INCLUDE_DIR})
    install(TARGETS dfttest2_cpu LIBRARY DESTINATION lib)
endif()

target_include_directories(dfttest2_cpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/..)


if(CPU_DISPATCH_TARGETS)
    target_compile_definitions(dfttest2_cpu PRIVATE HAS_DISPATCH)

    set(GETFRAME_DECLARATIONS "")
    set(GETFRAME_PTRS "")
    set(SUPPORTED_ARCH_DECLARATIONS "")
    set(SUPPORTED_ARCH_PTRS "")
    set(SUPPORTED_ARCH_STRS "")

    foreach(arch_option ${CPU_DISPATCH_TARGETS})
        set(raw_arch_option ${arch_option})
        string(REPLACE "=" "_" arch ${arch_option})
        string(REPLACE "-" "_" arch ${arch})

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
            if(${arch_option} STREQUAL avx2)
                set(arch_option ${arch_option} -mfma)
            elseif(${arch_option} STREQUAL avx512f)
                if(WIN32)
                    # according to vcl2,
                    # MS compiler cannot generate code for AVX512F without AVX512DQ
                    set(arch_option ${arch_option} -mfma -mavx512vl -mavx512bw -mavx512dq)
                else()
                    set(arch_option ${arch_option} -mfma)
                endif()
            endif()
        endif()

        set(current_target getframe_impl_${arch})
        set(current_getframe_name DFTTestGetFrame_${arch})
        set(current_supported_arch_name supported_arch_${arch})

        add_library(${current_target} OBJECT getframe_impl.cpp)

        set_target_properties(${current_target} PROPERTIES
            CXX_EXTENSIONS OFF
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )

        target_compile_definitions(${current_target} PRIVATE HAS_DISPATCH)
        target_compile_definitions(${current_target} PRIVATE DFTTEST_GETFRAME_NAME=${current_getframe_name})
        target_compile_definitions(${current_target} PRIVATE SUPPORTED_ARCH_NAME=${current_supported_arch_name})

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
            target_compile_options(${current_target} PRIVATE -m${arch_option})
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            target_compile_options(${current_target} PRIVATE /arch:${arch_option})
        endif()

        target_include_directories(${current_target} PRIVATE ${VCL_HOME})

        if(PKG_CONFIG_FOUND AND VS_FOUND)
            target_include_directories(${current_target} PRIVATE ${VS_INCLUDE_DIRS})
        else()
            target_include_directories(${current_target} PRIVATE ${VS_INCLUDE_DIR})
        endif()

        target_include_directories(${current_target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

        string(APPEND GETFRAME_DECLARATIONS "
        extern const VSFrameRef *VS_CC ${current_getframe_name}(
            int n, int activationReason, void **instanceData, void **frameData,
            VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
        ) noexcept;\n")

        string(APPEND GETFRAME_PTRS ${current_getframe_name},)

        string(APPEND SUPPORTED_ARCH_DECLARATIONS "extern bool ${current_supported_arch_name}() noexcept;\n")

        string(APPEND SUPPORTED_ARCH_PTRS ${current_supported_arch_name},)

        string(APPEND SUPPORTED_ARCH_STRS \"${raw_arch_option}\",)

        target_link_libraries(dfttest2_cpu PRIVATE ${current_target})
    endforeach()

    configure_file(cpu_dispatch.h.in cpu_dispatch.h @ONLY)

    target_compile_definitions(dfttest2_cpu PRIVATE GETFRAME_PTRS=${GETFRAME_PTRS})
    target_compile_definitions(dfttest2_cpu PRIVATE SUPPORTED_ARCH_PTRS=${SUPPORTED_ARCH_PTRS})
    target_compile_definitions(dfttest2_cpu PRIVATE SUPPORTED_ARCH_STRS=${SUPPORTED_ARCH_STRS})
    target_include_directories(dfttest2_cpu PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
else()
    add_library(getframe_impl OBJECT getframe_impl.cpp)

    set_target_properties(getframe_impl PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    target_include_directories(getframe_impl PRIVATE ${VCL_HOME})

    if(PKG_CONFIG_FOUND AND VS_FOUND)
        target_include_directories(getframe_impl PRIVATE ${VS_INCLUDE_DIRS})
    else()
        target_include_directories(getframe_impl PRIVATE ${VS_INCLUDE_DIR})
    endif()

    target_link_libraries(dfttest2_cpu PRIVATE getframe_impl)
endif()
