cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(rim-lab VERSION 0.1.0 LANGUAGES CXX CUDA)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_subdirectory(extern)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

add_executable(rim-lab
    src/main.cpp
    src/fractals.cu
)

target_include_directories(rim-lab
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${GLUT_INCLUDE_DIRS}
)

target_link_libraries(rim-lab
    PRIVATE
        OpenGL::OpenGL
        ${GLUT_LIBRARY}
        cuda-samples
)

set_target_properties(rim-lab
    PROPERTIES
        CXX_STANDARD 17
        CXX_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
)

# target_compile_options(rim-lab
#     PRIVATE
#         -Wall -Wextra -pedantic
# )
