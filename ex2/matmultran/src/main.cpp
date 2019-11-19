#include <cmath>
#include <cstdio>
#include <ctime>

#include <chrono>
#include <thread>

#include "matmultran.hpp"

void alloc_mem(int m, int n, float **A_ptr, float **C_ptr, float **D_ptr)
{
    *A_ptr = (float *) malloc(m * n * sizeof(float));
    *C_ptr = (float *) malloc(m * m * sizeof(float));
    *D_ptr = (float *) malloc(m * m * sizeof(float));
}

void free_mem(float *A, float *C, float *D)
{
    free(A);
    free(C);
    free(D);
}

void read_data(int *m_ptr, int *n_ptr, float **A_ptr, float **C_ptr, float **D_ptr)
{
    FILE *f = fopen("matmultran.dat", "rb");
    fread(m_ptr, sizeof(int), 1, f);
    fread(n_ptr, sizeof(int), 1, f);

    alloc_mem(*m_ptr, *n_ptr, A_ptr, C_ptr, D_ptr);

    fread(*A_ptr, sizeof(float), *m_ptr * *n_ptr, f);
    fread(*D_ptr, sizeof(float), *m_ptr * *m_ptr, f);

    fclose(f);
}

void matcmp(float *C, float *D, int m, int n)
{
    int k;
    float d, e = -1.0f;
    for (k = 0; k < m * n; k++)
    {
        if ((d = fabsf(C[k] - D[k])) > e)
        {
            e = d;
        }
    }

    printf("max. abs. err. = %.1e\n", e);
}

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
typedef LARGE_INTEGER app_timer_t;
#define timer(t_ptr) QueryPerformanceCounter(t_ptr)
void elapsed_time(app_timer_t start, app_timer_t stop, double flop)
{
    double etime;
    LARGE_INTEGER clk_freq;
    QueryPerformanceFrequency(&clk_freq);
    etime = (stop.QuadPart - start.QuadPart) / (double)clk_freq.QuadPart;
    printf("CPU (total!) time = %.3f ms (%6.3f GFLOP/s)\n", etime * 1e3, 1e-9 * flop / etime);
}
#else
using app_timer_t = std::chrono::time_point<std::chrono::steady_clock>;
#define timer(t_ptr) *t_ptr = std::chrono::steady_clock::now()
void elapsed_time(app_timer_t start, app_timer_t stop, double flop)
{
    const auto diff = stop - start;
    const auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff);
    const auto diff_ms_count = diff_ms.count();
    printf("CPU (total!) time = %ldms (%6.3f GFLOP/s)\n", diff_ms_count, flop/diff_ms_count);
}
#endif

int main(int argc, char *argv[])
{
    app_timer_t start, stop;
    int m, n;
    float *A, *C, *D;

    read_data(&m, &n, &A, &C, &D);

    timer(&start);
    matmultran(C, A, m, n);
    timer(&stop);
    elapsed_time(start, stop, 2 * m * m * n);

    matcmp(C, D, m, m);

    free_mem(A, C, D);

    return 0;
}
