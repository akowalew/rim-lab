#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h> //-I$(NVCUDASAMPLES_ROOT)/common/inc
#include <cublas_v2.h>   // wymaga konsolidacji z cublas.lib

#include <stdio.h>

#include "raindemo.hpp"

#define TRY(code) checkCudaErrors((cudaError_t) (code))

void csmp(float *xs, float *y, float *A, int N, int M, ...)
{
	float *dev_A, *dev_r, *dev_sp;
	cublasHandle_t h;
	float one = 1.0f, zero = 0.0f, nrm2y, nrm2a, nrm2r, s;
	int i, t;

	TRY(cudaSetDevice(0));

	TRY(cudaMalloc(&dev_A, M*N * sizeof(float)));
	TRY(cudaMalloc(&dev_r, M * sizeof(float)));
	TRY(cudaMalloc(&dev_sp, N * sizeof(float)));

	TRY(cublasCreate(&h));

	// TODO: prolog algorytmu

	for (t = 1, nrm2r = nrm2y; t <= 50 && nrm2r > 0.05*nrm2y; t++)
	{
		// TODO: iteracja algorytmu

		printf("iter.%3d: x(%3d) <- %4.2f, nrm2res=%4.2f\n",
			t, i, s, nrm2r);
	}

	TRY(cublasDestroy(h));

	TRY(cudaFree(dev_sp));
	TRY(cudaFree(dev_r));
	TRY(cudaFree(dev_A));

	TRY(cudaDeviceReset()); // dla debuggera i profilera
}
