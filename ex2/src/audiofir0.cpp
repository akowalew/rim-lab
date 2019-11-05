#include "audiofir.hpp"

static void audiofir_kernel(int i, float *yout, float *yin,
    float *coeff, int n, int len)
{
    /* Tu trzeba będzie wstawić prawdziwy kod... */
}

void audiofir(float *yout, float *yin,
    float *coeff, int n, int len,...)
{
    for (auto i = 0; i < len; i++)
    {
        audiofir_kernel(i, yout, yin, coeff, n, len);
    }

    for (auto i = 0; i < len; i++)
    {
        audiofir_kernel(i, yout+len, yin+len, coeff, n, len);
    }
}
