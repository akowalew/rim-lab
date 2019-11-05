###############################################################################
# CMakeLists.txt
#
# Defines structure of solution, needed dependencies
# and includes subdirectories
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 5.11.2019 20:32 CEST
###############################################################################

cmake_minimum_required(VERSION 3.10.0 FATAL_ERROR)
project(rim-lab VERSION 0.1.0 LANGUAGES CXX CUDA)

# Global definitions (same for all targets)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

# Find needed dependencies
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Load external modules
add_subdirectory(extern)

# Link globally (same for all targets)
link_libraries(cuda-samples)

# Define applications
add_subdirectory(ex1)
add_subdirectory(ex2)
