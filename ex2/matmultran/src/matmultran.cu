#include "matmultran.hpp"

#include <cassert>

#include <helper_cuda.h>

#define K 16 // rozmiar "kafelka"

__global__
static void matmultran_kernel(float *C, float *A, int m, int n)
{
    int tx = threadIdx.x; // kolumna wątku w ramach "kafelka"
    int ty = threadIdx.y; // wiersz wątku w ramach "kafelka"
    int ix = blockIdx.x * K + tx; // kolumna wątku w sieci
    int iy = blockIdx.y * K + ty; // wiersz wątku w sieci
    int iAT = blockIdx.x * K * n; // początek "kafelka" w A
    int iA = blockIdx.y * K * n; // początek "kafelka" w AT
    float s = 0;

    __shared__ float As[K][K], ATs[K][K];
    for(int t = 0; t < n / K; t++, iA += K, iAT += K)
    {
        As [ty][tx] = A[iA + ty*n + tx];
        ATs[ty][tx] = A[iAT + tx*n + ty];

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < K; k++)
        {
            s += As[ty][k] * ATs[k][tx];
        }

        __syncthreads();
    }

    C[iy*m + ix] = s;
}

void matmultran(float *C, float *A, int m, int n)
{
    checkCudaErrors(cudaSetDevice(0));

    float *dev_A, *dev_C;
    checkCudaErrors(cudaMalloc(&dev_A, m*n*sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_C, m*m*sizeof(float)));
    checkCudaErrors(cudaMemcpy(dev_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimGrid(m/K, m/K), dimBlock(K, K);

    cudaEvent_t start, stop; // pomiar czasu wykonania jądra
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    matmultran_kernel<<<dimGrid, dimBlock>>>(dev_C, dev_A, m, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventRecord(stop, 0));

    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime,start, stop));

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(C, dev_C, m*m*sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dev_C));
    checkCudaErrors(cudaFree(dev_A));

    checkCudaErrors(cudaDeviceReset()); // dla debuggera

    printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n", elapsedTime, 2e-6 * m * m * n / elapsedTime);
}
