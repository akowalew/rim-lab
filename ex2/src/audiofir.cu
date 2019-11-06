#include "audiofir.hpp"

#include <cassert>

#include <helper_cuda.h>

__global__
static void audiofir_kernel(float *yout, float *yin,
    float *coeff, int n, int len)
{
    assert(yout != nullptr);
    assert(yin != nullptr);
    assert(coeff != nullptr);
    assert(n < len);

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < len)
    {
        return;
    }

    auto j = i;
    auto sum = 0.0f;
    for(auto k = 0; k <= n; ++k, --j)
    {
        if(j < 0)
        {
            // For non-existent samples, tract them as zero
            // So there is no need to make further sums
            break;
        }

        sum += (yin[j] * coeff[k]);
    }

    yout[i] = sum;
}

void audiofir(float *yout, float *yin,
    float *coeff, int n, int len)
{
    checkCudaErrors(cudaSetDevice(0));

    float* d_yout;
    float* d_yin;
    float* d_coeff;

    checkCudaErrors(cudaMalloc(&d_yout, sizeof(float) * 2 * len));
    checkCudaErrors(cudaMalloc(&d_yin, sizeof(float) * 2 * len));
    checkCudaErrors(cudaMalloc(&d_coeff, sizeof(float) * (n + 1)));

    assert(d_yout != nullptr);
    assert(d_yin != nullptr);
    assert(d_coeff != nullptr);

    checkCudaErrors(cudaMemcpy(d_yin, yin, sizeof(float) * 2 * len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_coeff, coeff, sizeof(float) * (n + 1), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop; // pomiar czasu wykonania jądra
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    constexpr auto K = 512;
    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout, d_yin, d_coeff, n, len);
    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout+len, d_yin+len, d_coeff, n, len);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(yout, d_yout, sizeof(float) * 2 * len, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_yin));
    checkCudaErrors(cudaFree(d_yout));
    checkCudaErrors(cudaFree(d_coeff));

    checkCudaErrors(cudaDeviceReset());

    printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n",
            elapsedTime, 1e-6 * 2*((double)n+1) * 2*((double)len) / elapsedTime);
}
