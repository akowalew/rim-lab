###############################################################################
# CMakeLists.txt
#
# Defines build structure for application `ex2_matmultran` application from
#  the excercise 2
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 19.11.2019 21:18 CEST
###############################################################################

add_executable(ex2_matmultran
    src/matmultran.cu
    src/main.cpp
)

target_include_directories(ex2_matmultran
    PRIVATE
        include
)
