#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand.h>

#include <thrust/functional.h> // function objects & tools
#include <thrust/random.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#ifdef _WIN32

# define WINDOWS_LEAN_AND_MEAN
#include <windows.h>

typedef LARGE_INTEGER app_timer_t;
static inline void timer(app_timer_t *t_ptr)
{
#ifdef __CUDACC__
	checkCudaErrors(cudaDeviceSynchronize());
#endif
	QueryPerformanceCounter(t_ptr);
}

double elapsed_time(app_timer_t start, app_timer_t stop)
{
	LARGE_INTEGER clk_freq;
	QueryPerformanceFrequency(&clk_freq);
	return (stop.QuadPart - start.QuadPart) / (double) clk_freq.QuadPart * 1e3;
}

#else // defined(_WIN32)

#include <time.h>

typedef struct timespec app_timer_t;
static inline void timer(app_timer_t *t_ptr)
{
#ifdef __CUDACC__
	checkCudaErrors(cudaDeviceSynchronize());
#endif
	clock_gettime(CLOCK_MONOTONIC, t_ptr);
}

double elapsed_time(app_timer_t start, app_timer_t stop)
{
	return 1e+3 * (stop.tv_sec - start.tv_sec ) + 1e-6 * (stop.tv_nsec - start.tv_nsec);
}

#endif // defined(_WIN32)

class randuni
	:	public thrust::unary_function<unsigned long long, float>
{
private:
	thrust::default_random_engine rng;
	thrust::uniform_real_distribution<float> uni;

public:
	randuni(unsigned int seed, float a=0.0f, float b=1.0f)
		:	rng(seed)
		,	uni(a, b)
	{}

	__host__ __device__
	float operator()(unsigned long long i)
	{
		rng.discard(i); // odrzuć liczby z "poprzednich" wątków
		return uni(rng);
	}
};

class uniformAB
    :   public thrust::unary_function<float, float>
{
private:
    float a, b;

public:
    uniformAB(float _a, float _b)
        :   a(_a)
        ,   b(_b)
    {}

    __host__ __device__
    float operator()(float x) const
    {
        return x * (b - a) + a;
    }
};

using point3D = thrust::tuple<float, float, float>;

struct fun
	: public thrust::unary_function<point3D, float>
{
	__host__ __device__
	float operator ()(const point3D &p) const
	{
		const auto x = thrust::get<0>(p);
		const auto y = thrust::get<1>(p);
		const auto z = thrust::get<2>(p);
		const auto sum = x*x + y*y + z*z;

		return (sum > 1) ? 0.0f : 1.0f;
	}
};

#include <cstdlib>

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: ex2 <N>\n");
		return -1;
	}

	const auto N = atoi(argv[1]);

	app_timer_t t0, t1, t2, t3;
	timer(&t0); //--------------------------------------------

	thrust::device_vector<float> x(N);
	thrust::device_vector<float> y(N);
	thrust::device_vector<float> z(N);
	thrust::device_vector<float> xyz(3 * N);
	timer(&t1); //--------------------------------------------

    curandGenerator_t xyz_gen;
    curandCreateGenerator(&xyz_gen, CURAND_RNG_QUASI_DEFAULT);
	curandSetQuasiRandomGeneratorDimensions(xyz_gen, 3);
    curandGenerateUniform(xyz_gen, xyz.data().get(), xyz.size());
	curandDestroyGenerator(xyz_gen);

	thrust::copy(xyz.begin(), xyz.begin() + N, x.begin());
	thrust::copy(xyz.begin() + N, xyz.begin() + 2 * N, y.begin());
	thrust::copy(xyz.begin() + 2 * N, xyz.end(), z.begin());

    const auto uni = uniformAB(-1.0, 1.0);
    thrust::transform(x.begin(), x.end(), x.begin(), uni);
    thrust::transform(y.begin(), y.end(), y.begin(), uni);
    thrust::transform(z.begin(), z.end(), z.begin(), uni);

	timer(&t2); //--------------------------------------------

	// Numeric integral using Monte-Carlo algorithm
	const auto zip_begin_tuple = thrust::make_tuple(x.begin(), y.begin(), z.begin());
	const auto zip_begin = thrust::make_zip_iterator(zip_begin_tuple);

	const auto zip_end_tuple = thrust::make_tuple(x.end(), y.end(), z.end());
	const auto zip_end = thrust::make_zip_iterator(zip_end_tuple);

	const auto sum = thrust::transform_reduce(zip_begin, zip_end, fun(), 0.0f, thrust::plus<float>());
	const auto integral = (sum / N) * 8.0f;
	timer(&t3); //--------------------------------------------

	std::cout << "pi = " << 0.75f * integral << '\n';
	std::cout << "Initialization: " << elapsed_time(t0, t1) << " ms" << '\n';
	std::cout << "Generation: " << elapsed_time(t1, t2) << " ms" << '\n';
	std::cout << "Integral: " << elapsed_time(t2, t3) << " ms" << '\n';
	std::cout << "T O T A L : " << elapsed_time(t0, t3) << " ms" << '\n';

	#ifdef _WIN32
	if (IsDebuggerPresent()) getchar();
	#endif

	return 0;
}
