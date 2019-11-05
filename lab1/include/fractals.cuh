#pragma once

constexpr auto DIM = 1000; /* rozmiar rysunku w pikselach */
constexpr auto DIM_BLOCK = 16;

namespace fractals {

void init();

void cleanup();

void compute_julia(unsigned char* pixbuf, const float dx, const float dy, const float scale);

} // namespace fractals
