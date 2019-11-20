#define _CRT_SECURE_NO_WARNINGS
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "raindemo.h"

/**********************************************************/

void alloc_mem(int M, int N,
	float ** x_ptr, float ** A_ptr, float **y_ptr,
	float **xr_ptr, float **xs_ptr)
{
	*x_ptr = (float *)malloc(N * sizeof(float));
	*A_ptr = (float *)malloc(M * N * sizeof(float));
	*y_ptr = (float *)malloc(M * sizeof(float));
	*xr_ptr = (float *)malloc(N * sizeof(float));
	*xs_ptr = (float *)calloc(N, sizeof(float));
}

void free_mem(float *x, float *A, float *y,
	float *xr, float *xs)
{
	free(x);
	free(A);
	free(y);
	free(xr);
	free(xs);
}

void read_data(int    * M_ptr, int    * N_ptr,
	float ** x_ptr, float ** A_ptr, float **y_ptr,
	float **xr_ptr, float **xs_ptr)
{
	FILE *f = fopen("raindemo.dat", "rb");

	fread(M_ptr, sizeof(int), 1, f);
	fread(N_ptr, sizeof(int), 1, f);

	alloc_mem(*M_ptr, *N_ptr,
		x_ptr, A_ptr, y_ptr, xr_ptr, xs_ptr);

	fread(*x_ptr, sizeof(float), *N_ptr, f);
	fread(*A_ptr, sizeof(float), *M_ptr * *N_ptr, f);
	fread(*y_ptr, sizeof(float), *M_ptr, f);
	fread(*xr_ptr, sizeof(float), *N_ptr, f);

	fclose(f);
}

/**********************************************************/

void sigcmp(float *xs, float *xr, int N)
{
	int k;
	float d, e = -1.0f;
	for (k = 0; k < N; k++)
		if ((d = fabsf(xs[k] - xr[k])) > e)
			e = d;
	printf("max. abs. err. = %.1e\n", e);
}

/**********************************************************/

int main(int argc, char *argv[])
{
	int        M, N;
	float             *x, *A, *y, *xr, *xs;
	read_data(&M, &N, &x, &A, &y, &xr, &xs);
	csmp(xs, y, A, N, M, argc, argv);
	sigcmp(xs, xr, N);
	free_mem(x, A, y, xr, xs);
	if (IsDebuggerPresent()) getchar();
	return 0;
}