###############################################################################
# CMakeLists.txt
#
# Defines build structure for applications in excercise 3
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 20.11.2019 2:48 CEST
###############################################################################

option(BUILD_EX3_MCINT "" ON)
option(BUILD_EX3_RAINDEMO "" ON)

add_subdirectory(mcint)
add_subdirectory(raindemo)
