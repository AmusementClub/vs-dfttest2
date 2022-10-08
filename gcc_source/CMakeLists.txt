add_library(dfttest2_gcc MODULE source.cpp)

set_target_properties(dfttest2_gcc PROPERTIES
    CXX_EXTENSIONS OFF
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(dfttest2_gcc PRIVATE ${VCL_HOME})

if(PKG_CONFIG_FOUND AND VS_FOUND)
    target_include_directories(dfttest2_gcc PRIVATE ${VS_INCLUDE_DIRS})
    install(TARGETS dfttest2_gcc LIBRARY DESTINATION ${install_dir})
else()
    target_include_directories(dfttest2_gcc PRIVATE ${VS_INCLUDE_DIR})
    install(TARGETS dfttest2_gcc LIBRARY DESTINATION lib)
endif()

target_include_directories(dfttest2_gcc PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/..)


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

target_link_libraries(dfttest2_gcc PRIVATE getframe_impl)
