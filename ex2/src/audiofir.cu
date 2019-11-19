#include "audiofir.hpp"

#include <cassert>

#include <helper_cuda.h>

static constexpr int N = 1024; // Rząd filtru FIR
static constexpr int K = 512; // Ilość wątków w bloku

__constant__ static float fir_coeff[N + 1];

__global__
static void audiofir_kernel(float *yout, const float *yin)
{
    assert(yout != nullptr);
    assert(yin != nullptr);
	assert(threadIdx.x < K);

	__shared__ float ytile[N + K];

    // From where we start
    const int y_pos = (threadIdx.x + (blockIdx.x * blockDim.x));

    // Copy data from input vector to shared memory (aka 'ytile')
    int i = y_pos; // Position in input vector
    int j = N + threadIdx.x; // Position in tile
    #pragma unroll
    while(j >= 0)
    {
        ytile[j] = yin[i];

        i -= K;
        j -= K;
    }

    // Wait for data copy end
    __syncthreads();

    // Perform scalar multiply on fetched tile
    auto sum = 0.0f; // Scalar multiply sum
    int k = 0; // Position in coefficients
    j = (N + threadIdx.x); // Position in tile
    #pragma unroll
    while(k <= N)
    {
        const auto ytile_elem = ytile[j];
        const auto coeff_elem = fir_coeff[k];
        sum += (ytile_elem * coeff_elem);

        --j;
        ++k;
    }

    // Save calculated scalar
    yout[y_pos] = sum;
}

void audiofir(float *yout, const float *yin,
    const float *coeff, int n, int len)
{
	assert(n == N);

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

    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout, d_yin + n);
    audiofir_kernel<<<(len + K - 1) / K, K>>>(d_yout + len_1, d_yin + n + len_1 + n);
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
