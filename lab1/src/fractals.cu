#include "fractals.cuh"

#include <cstdio>
#include <cmath>

#include <cuda_runtime.h>

#include "helper_cuda.h"
#include "helper_timer.h"

namespace fractals {

namespace {

StopWatchInterface* timer = NULL;

constexpr auto cr = -0.123;
constexpr auto ci = 0.745;

unsigned char *d_pixbuf;

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

__global__ void julia_kernel(unsigned char *ptr, const float dx, const float dy, const float scale)
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

} // namespace

void init()
{
    checkCudaErrors(cudaSetDevice(0));
    checkCudaErrors(cudaMalloc(&d_pixbuf, 4 * DIM * DIM));
    sdkCreateTimer(&timer);
}

void cleanup()
{
    sdkDeleteTimer(&timer);
    checkCudaErrors(cudaFree(d_pixbuf));
    checkCudaErrors(cudaDeviceReset());
}

void compute_julia(unsigned char* pixbuf, const float dx, const float dy, const float scale)
{
    const auto dim_grid_x = (DIM + DIM_BLOCK - 1) / DIM_BLOCK;
    const auto dim_grid_y = (DIM + DIM_BLOCK - 1) / DIM_BLOCK;
    const auto dim_grid = dim3{dim_grid_x, dim_grid_y};
    const auto dim_block = dim3{DIM_BLOCK, DIM_BLOCK};

    sdkStartTimer(&timer);

    julia_kernel<<<dim_grid, dim_block>>>(d_pixbuf, dx, dy, scale);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(pixbuf, d_pixbuf, 4 * DIM * DIM, cudaMemcpyDeviceToHost));

    sdkStopTimer(&timer);
    printf("Processing time: %f ms\n", sdkGetTimerValue(&timer));
    sdkResetTimer(&timer);
}

} // namespace fractals
