###############################################################################
# CMakeLists.txt
#
# Defines build structure for application `ex2`
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 5.11.2019 20:37 CEST
###############################################################################

add_executable(ex2
    src/audiofir0.cpp
    src/main.cpp
)

target_include_directories(ex2
    PRIVATE
        include
)

configure_file(scripts/testfir.m testfir.m COPYONLY)

