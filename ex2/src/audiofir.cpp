#include "audiofir.hpp"

#include <cassert>

static void audiofir_kernel(int i, float *yout, float *yin,
    float *coeff, int n, int len)
{
    assert(yout != nullptr);
    assert(yin != nullptr);
    assert(coeff != nullptr);
    assert(i < len);
    assert(n < len);

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
    for (auto i = 0; i < len; i++)
    {
        audiofir_kernel(i, yout, yin, coeff, n, len);
    }

    for (auto i = 0; i < len; i++)
    {
        audiofir_kernel(i, yout+len, yin+len, coeff, n, len);
    }
}
