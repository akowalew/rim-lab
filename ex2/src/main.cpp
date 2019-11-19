#include <cmath>
#include <cstdio>
#include <ctime>

#include <chrono>
#include <thread>

#include "audiofir.hpp"

void alloc_mem(int n, int len,
    float **coeff_ptr, float **yin_ptr,
    float **yref_ptr, float **yout_ptr)
{
    *coeff_ptr = (float *) malloc((1 + n) * sizeof(float));
    *yin_ptr = (float *) malloc(2 * len * sizeof(float));
    *yref_ptr = (float *) malloc(2 * len * sizeof(float));
    *yout_ptr = (float *) malloc(2 * len * sizeof(float));
}

void free_mem(float *coeff_ptr, float *yin_ptr, float *yref_ptr, float *yout_ptr)
{
    free(coeff_ptr);
    free(yin_ptr);
    free(yref_ptr);
    free(yout_ptr);
}

void read_data(int *n_ptr, int *len_ptr,
    float **coeff_ptr, float **yin_ptr,
    float **yref_ptr, float **yout_ptr)
{
    auto file = fopen("audiofir_in.dat", "rb");
    fread(n_ptr, sizeof(int), 1, file);
    fread(len_ptr, sizeof(int), 1, file);

    alloc_mem(*n_ptr, *len_ptr, coeff_ptr, yin_ptr, yref_ptr, yout_ptr);

    fread(*coeff_ptr, sizeof(float), 1 + *n_ptr, file);
    fread(*yin_ptr, sizeof(float), 2 * *len_ptr, file);
    fread(*yref_ptr, sizeof(float), 2 * *len_ptr, file);

    fclose(file);
}

void write_data(int len, float *y)
{
    auto file = fopen("audiofir_out.dat", "wb");
    fwrite(y, sizeof(float), 2 * len, file);
    fclose(file);
}

void audiocmp(float *yout, float *yref, int len)
{
    auto e = -1.0f;
    for(auto k = 0; k < 2 * len; k++)
    {
        if(const auto d = fabsf(yout[k] - yref[k]); d > e)
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
    int n, len;
    float *coeff, *yin, *yref, *yout;

    read_data(&n, &len, &coeff, &yin, &yref, &yout);

    timer(&start);
    audiofir(yout, yin, coeff, n, len);
    timer(&stop);
    elapsed_time(start, stop, 2*((double)n+1) * 2*((double)len));

    audiocmp(yout, yref, len);
    write_data(len, yout);

    free_mem(coeff, yin, yref, yout);
}
