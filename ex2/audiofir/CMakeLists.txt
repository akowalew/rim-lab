###############################################################################
# CMakeLists.txt
#
# Defines build structure for application `audiofir` application from
#  the excercise 2
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 19.11.2019 21:17 CEST
###############################################################################

add_executable(ex2_audiofir
    src/audiofir.cu
    src/main.cpp
)

target_include_directories(ex2_audiofir
    PRIVATE
        include
)

configure_file(scripts/testfir.m testfir.m COPYONLY)
configure_file(scripts/audiofir.m audiofir.m COPYONLY)

target_compile_options(ex2_audiofir
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)
