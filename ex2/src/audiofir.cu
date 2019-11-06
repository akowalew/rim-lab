#include "audiofir.hpp"

#include <cassert>

#include <helper_cuda.h>

static constexpr int N = 1024; // maksymalny rząd filtru FIR 
static constexpr int K = 512; // Ilość wątków w bloku

__constant__ static float fir_coeff[N + 1];

__global__
static void audiofir_kernel(float *yout, const float *yin, int n)
{
    assert(yout != nullptr);
    assert(yin != nullptr);
	assert(n <= N);
	assert(threadIdx.x < K);

	__shared__ float ytile[N + K];

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
	for (int j = i; j >= (i - n); j -= K)
	{
		ytile[j] = yin[i];
	}



	yin += i;
	yout += i;

	float* coeff = fir_coeff;
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
	assert(n <= N);

    checkCudaErrors(cudaSetDevice(0));

    float* d_yout;
    float* d_yin;

    const unsigned int len_1 = (K * ((len + K - 1) / K));

    checkCudaErrors(cudaMalloc(&d_yout, sizeof(float) * 2 * len_1));
    checkCudaErrors(cudaMalloc(&d_yin, sizeof(float) * 2 * (len_1 + n)));

    assert(d_yout != nullptr);
    assert(d_yin != nullptr);

	checkCudaErrors(cudaMemset(d_yin, 0, sizeof(float) * 2 * (len_1 + n)));
    checkCudaErrors(cudaMemcpy(d_yin + n, yin, sizeof(float) * len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_yin + n + len_1 + n, yin + len, sizeof(float) * len, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpyToSymbol(fir_coeff, coeff, sizeof(float) * (n + 1)));

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEvent_t start, stop; // pomiar czasu wykonania jądra
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout, d_yin + n, n);
    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout + len_1, d_yin + n + len_1 + n, n);
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

    checkCudaErrors(cudaDeviceReset());

    printf("GPU (kernel) time = %.3f ms (%6.3f GFLOP/s)\n",
            elapsedTime, 1e-6 * 2*((double)n+1) * 2*((double)len) / elapsedTime);
}
