add_library(cuda-samples INTERFACE)

target_include_directories(cuda-samples
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda-samples/Common
)
