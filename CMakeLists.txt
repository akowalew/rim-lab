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

option(USE_CUDA "Whether to compile CUDA programs or not" ON)
option(BUILD_EX1 "Whether to build excercise 1 or not" ON)
option(BUILD_EX2 "Whether to build excercise 2 or not" ON)
option(BUILD_EX3 "Whether to build excercise 3 or not" ON)

project(rim-lab VERSION 0.1.0 LANGUAGES CXX)

if(USE_CUDA)
	enable_language(CUDA)
endif()

# Global definitions (same for all targets)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")

# Load external modules
add_subdirectory(extern)

# Link globally (same for all targets)
link_libraries(cuda-samples)

# Define applications
if(BUILD_EX1)
	add_subdirectory(ex1)
endif()

if(BUILD_EX2)
	add_subdirectory(ex2)
endif()

if(BUILD_EX3)
	add_subdirectory(ex3)
endif()
