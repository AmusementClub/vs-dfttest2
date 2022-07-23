cmake_minimum_required(VERSION 3.22.0)

project(vs-dfttest2 VERSION 0.1 LANGUAGES CXX)

set(USE_NVRTC_STATIC ON CACHE BOOL "Whether to use NVRTC static library")

find_package(CUDAToolkit REQUIRED)

add_library(dfttest2 SHARED source/source.cpp)

set_target_properties(dfttest2 PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(dfttest2 PRIVATE CUDA::cuda_driver CUDA::cufft)

if(
    USE_NVRTC_STATIC AND (
        CUDAToolkit_VERSION_MAJOR GREATER_EQUAL "12" OR (
            CUDAToolkit_VERSION_MAJOR EQUAL "11" AND
            CUDAToolkit_VERSION_MINOR GREATER_EQUAL "5"
        )
    )
)
    target_link_directories(dfttest2 PRIVATE "${CUDAToolkit_LIBRARY_DIR}")
    target_link_libraries(dfttest2 PRIVATE nvrtc_static nvrtc-builtins_static nvptxcompiler_static)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set_property(TARGET dfttest2 PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        target_link_libraries(dfttest2 PRIVATE Ws2_32)
    endif()
else()
    if(USE_NVRTC_STATIC)
        message(WARNING "NVRTC static library is not used")
    endif()
    target_link_libraries(dfttest2 PRIVATE CUDA::nvrtc)
endif()

find_package(PkgConfig QUIET MODULE)

if(PKG_CONFIG_FOUND)
    pkg_search_module(VS vapoursynth)

    if(VS_FOUND)
        message(STATUS "Found VapourSynth r${VS_VERSION}")

        cmake_path(APPEND install_dir ${VS_LIBDIR} vapoursynth)
        target_include_directories(dfttest2 PRIVATE ${VS_INCLUDE_DIRS})

        install(TARGETS dfttest2 LIBRARY DESTINATION ${install_dir})
    endif()
endif()

if(NOT VS_FOUND)
    set(VS_INCLUDE_DIR "" CACHE PATH "Path to VapourSynth headers")

    if(VS_INCLUDE_DIR STREQUAL "")
        message(WARNING "VapourSynth not found")
    endif()

    target_include_directories(dfttest2 PRIVATE ${VS_INCLUDE_DIR})

    install(TARGETS dfttest2 LIBRARY RUNTIME)
endif()
