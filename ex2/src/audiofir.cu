#include "audiofir.hpp"

#include <cassert>

#include <helper_cuda.h>

__global__
static void audiofir_kernel(float *yout, const float *yin,
    const float *coeff, int n)
{
    assert(yout != nullptr);
    assert(yin != nullptr);
    assert(coeff != nullptr);

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
	yin += i;
	yout += i;

    float sum = 0.0f;
    for(int k = 0; k <= n; ++k)
    {
        const float yin_elem = *(yin--);
        const float coeff_elem = *(coeff++);
        sum += (yin_elem * coeff_elem);
    }

    *yout = sum;
}

void audiofir(float *yout, const float *yin,
    const float *coeff, int n, int len)
{
    checkCudaErrors(cudaSetDevice(0));

    float* d_yout;
    float* d_yin;
    float* d_coeff;

    static constexpr int K = 512;
    const unsigned int len_1 = (K * ((len + K - 1) / K));
    printf("%d %d\n", len, len_1);

    checkCudaErrors(cudaMalloc(&d_yout, sizeof(float) * 2 * len_1));
    checkCudaErrors(cudaMalloc(&d_yin, sizeof(float) * 2 * (len_1 + n)));
    checkCudaErrors(cudaMalloc(&d_coeff, sizeof(float) * (n + 1)));

    assert(d_yout != nullptr);
    assert(d_yin != nullptr);
    assert(d_coeff != nullptr);

	checkCudaErrors(cudaMemset(d_yin, 0, sizeof(float) * 2 * (len_1 + n)));
    checkCudaErrors(cudaMemcpy(d_yin + n, yin, sizeof(float) * len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_yin + n + len_1 + n, yin + len, sizeof(float) * len, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_coeff, coeff, sizeof(float) * (n + 1), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop; // pomiar czasu wykonania jądra
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout, d_yin + n, d_coeff, n);
    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout + len_1, d_yin + n + len_1 + n, d_coeff, n);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float elapsedTime;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(yout, d_yout, sizeof(float) * len, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(yout+len, d_yout+len_1, sizeof(float) * len, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_yin));
    checkCudaErrors(cudaFree(d_yout));
    checkCudaErrors(cudaFree(d_coeff));

    checkCudaErrors(cudaDeviceReset());

    printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n",
            elapsedTime, 1e-6 * 2*((double)n+1) * 2*((double)len) / elapsedTime);
}
