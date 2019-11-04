#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>

#include <GL/glut.h>

#include "helper_cuda.h"
#include "helper_timer.h"

constexpr auto cr = -0.123;
constexpr auto ci = 0.745;

constexpr auto DIM = 1000; /* rozmiar rysunku w pikselach */
constexpr auto DIM_BLOCK = 16;

StopWatchInterface* timer = NULL;

__forceinline__ __device__ int julia(float x, float y)
{
    const auto abs_c = std::hypot(ci, cr);
    const auto R_2 = (abs_c > 2) ? (abs_c * abs_c) : 4;
    constexpr auto N = 40;
    for(auto i = 0; i < N; ++i)
    {
        auto x_2 = x * x;
        auto y_2 = y * y;
        if(x_2 + y_2 > R_2)
        {
            // Uciekinierzy
            return 0;
        }

        y = 2 * x * y + ci;
        x = x_2 - y_2 + cr;
    }

    // Więżniowie
    return 1;
}

__global__ void kernel(unsigned char *ptr,
    const float dx, const float dy,
    const float scale)
{
    const int xw = blockIdx.x * blockDim.x + threadIdx.x;
    const int yw = blockIdx.y * blockDim.y + threadIdx.y;
    if(xw > DIM || yw > DIM)
    {
        return;
    }

    /* przeliczenie współrzędnych pikselowych (xw, yw)
    na matematyczne (x, y) z uwzględnieniem skali
    (scale) i matematycznego środka (dx, dy) */
    const auto x = scale * (xw - DIM/2) / (DIM/2) + dx;
    const auto y = scale * (yw - DIM/2) / (DIM/2) + dy;
    const auto offset /* w buforze pikseli */ = xw + yw*DIM;

    /* kolor: czarny dla uciekinierów (julia == 0)
    czerwony dla więźniów (julia == 1) */
    ptr[offset*4 + 0 /* R */] = (unsigned char) (255*julia(x,y));
    ptr[offset*4 + 1 /* G */] = 0;
    ptr[offset*4 + 2 /* B */] = 0;
    ptr[offset*4 + 3 /* A */] = 255;
}
/**************************************************/

static unsigned char *d_pixbuf;
static unsigned char pixbuf[DIM * DIM * 4];
static float dx = 0.0f, dy = 0.0f;
static float scale = 1.5f;

static void disp(void)
{
    glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, pixbuf);
    glutSwapBuffers();
}

static void recompute(void)
{
    const auto dim_grid_x = (DIM + DIM_BLOCK - 1) / DIM_BLOCK;
    const auto dim_grid_y = (DIM + DIM_BLOCK - 1) / DIM_BLOCK;
    const auto dim_grid = dim3{dim_grid_x, dim_grid_y};
    const auto dim_block = dim3{DIM_BLOCK, DIM_BLOCK};

    sdkStartTimer(&timer);

    kernel<<<dim_grid, dim_block>>>(d_pixbuf, dx, dy, scale);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(pixbuf, d_pixbuf, sizeof(pixbuf), cudaMemcpyDeviceToHost));

    sdkStopTimer(&timer);
    printf("Processing time: %f ms\n", sdkGetTimerValue(&timer));
    sdkResetTimer(&timer);

    glutPostRedisplay();
}

static void kbd(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'p':
            dx += scale * (x - DIM/2) / (DIM/2);
            break;
        case 'o':
            dx -= scale * (x - DIM/2) / (DIM/2);
            break;
        case 'i':
            dy -= scale * (y - DIM/2) / (DIM/2);
            break;
        case 'u':
            dy += scale * (y - DIM/2) / (DIM/2);
            break;
        case 'z':
            scale *= 0.80f;
            break;
        case 'Z':
            scale *= 1.25f;
            break;
        case '=':
            scale = 1.50f;
            dx = dy = 0.0f;
            break;
        case 27:
            /* Esc */ exit(0);
    }

    recompute();
}

int main(int argc, char *argv[])
{
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMalloc(&d_pixbuf, 4 * DIM * DIM));
    sdkCreateTimer(&timer);

    glutInit(&argc, argv); /* inicjacja biblioteki GLUT */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA); /* opcje */
    glutInitWindowSize(DIM, DIM); /* rozmiar okna graficznego */
    glutCreateWindow("RIM - fraktal Julii"); /* tytuł okna */
    glutDisplayFunc(disp); /* funkcja zwrotna zobrazowania */
    glutKeyboardFunc(kbd); /* funkcja zwrotna klawiatury */
    recompute(); /* obliczenie pierwszego rysunku */
    glutMainLoop(); /* główna pętla obsługi zdarzeń */

    sdkDeleteTimer(&timer);
    checkCudaErrors(cudaFree(d_pixbuf));
    checkCudaErrors(cudaDeviceReset());
}
