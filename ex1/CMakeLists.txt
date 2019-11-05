###############################################################################
# CMakeLists.txt
#
# Defines build structure for application `ex1`
#
# Author: akowalew (ram.techen@gmail.com)
# Date: 5.11.2019 20:42 CEST
###############################################################################

add_executable(ex1
    src/main.cpp
    src/fractals.cu
)

target_include_directories(ex1
    PRIVATE
        include
        ${GLUT_INCLUDE_DIRS}
)

target_link_libraries(ex1
    PRIVATE
        OpenGL::OpenGL
        ${GLUT_LIBRARY}
)
