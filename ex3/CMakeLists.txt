###############################################################################
# CMakeLists.txt
#
# Defines build structure for application `ex3`
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 20.11.2019 00:50 CEST
###############################################################################

add_executable(ex3
    src/main.cpp
)

target_include_directories(ex3
    PRIVATE
        include
)

target_compile_options(ex3
	PRIVATE
		-fopenmp
)

target_link_libraries(ex3
	PRIVATE
		-fopenmp
)
