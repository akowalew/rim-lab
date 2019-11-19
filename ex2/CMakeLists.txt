###############################################################################
# CMakeLists.txt
#
# Defines build structure for application `ex2`
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 5.11.2019 20:37 CEST
###############################################################################

add_executable(ex2
    src/audiofir.cu
    src/main.cpp
)

target_include_directories(ex2
    PRIVATE
        include
)

configure_file(scripts/testfir.m testfir.m COPYONLY)
configure_file(scripts/audiofir.m audiofir.m COPYONLY)

target_compile_options(ex2
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
