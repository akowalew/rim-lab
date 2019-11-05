#include <cmath>
#include <cstdio>
#include <ctime>

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
    free( yin_ptr);
    free( yref_ptr);
    free( yout_ptr);
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

using app_timer_t = timespec;

#define timer(t_ptr) clock_gettime(CLOCK_MONOTONIC, t_ptr)

void elapsed_time(app_timer_t start, app_timer_t stop, double flop)
{
    const auto sec_diff = (stop.tv_sec - start.tv_sec);
    const auto nsec_diff = (stop.tv_nsec - start.tv_nsec);
    const auto etime = 1e+3 * sec_diff + 1e-6 * nsec_diff;
    printf("CPU (total!) time = %.3f ms (%6.3f GFLOP/s)\n",
        etime, 1e-6 * flop / etime);
}

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
