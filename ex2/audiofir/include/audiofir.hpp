#pragma once

void audiofir(float *yout, const float *yin,
    const float *coeff, int n, int len);
