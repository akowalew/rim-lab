###############################################################################
# CMakeLists.txt
#
# Defines build structure for application raindemo
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 20.11.2019 3:04 CEST
###############################################################################

configure_file(scripts/raindemo.m raindemo.m COPYONLY)

add_executable(raindemo
	src/main.cpp
	src/raindemo.cu
)

target_include_directories(raindemo
	PRIVATE
		include
)