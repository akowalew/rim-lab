###############################################################################
# CMakeLists.txt
#
# Defines build structure for application `mcint`
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 20.11.2019 00:50 CEST
###############################################################################

add_executable(mcint
    src/main.cu
)

target_link_libraries(mcint
    PRIVATE
        curand
)
