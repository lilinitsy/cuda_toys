#include <algorithm>
#include <chrono>
#include <iostream>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

const size_t SIZE = 2e6; // 1,000,000



struct RandomFloatGenerator
{
	RandomFloatGenerator()
	{
		srand(time(0));
	}

	float operator()() const
	{
		return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
};



float cpu_median(float *arr, size_t size)
{
	size_t midpoint = size / 2;

	std::nth_element(arr, arr + midpoint, arr + size);

	if(size & 0b01 == 0)
	{
		float mid1 = arr[midpoint];
		std::nth_element(arr, arr + midpoint - 1, arr + size);
		float mid2 = arr[midpoint - 1];
		return (mid1 + mid2) / 2.0f;
	}

	return arr[midpoint];
}

float *generate_array(size_t size)
{
	float *data = new float[size];

	for(size_t i = 0; i < size; i++)
	{
		float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		data[i] = r;
	}

	return data;
}

__host__ float thrust_find_median(thrust::device_vector<float> &device_vec)
{
	thrust::sort(thrust::device, device_vec.begin(), device_vec.end());

	size_t midpoint = device_vec.size() / 2;
	float  mid1     = device_vec[midpoint];

	if(device_vec.size() & 0b01 == 0)
	{
		float mid2 = device_vec[midpoint - 1];
		return (mid1 + mid2) / 2.0f;
	}

	return mid1;
}



int main()
{
	srand(time(0));

	// Time data creation
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

	float *data = generate_array(SIZE);

	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

	std::chrono::milliseconds dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	std::chrono::microseconds dt_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("CPU Time to generate data: %lu ms %lu us\n", dt_ms.count(), dt_us.count());

	// Time finding median
	start_time = std::chrono::high_resolution_clock::now();

	float median = cpu_median(data, SIZE);

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("CPU Median: %f\nTime to find CPU median: %lu ms %lu us\n", median, dt_ms.count(), dt_us.count());

	// Cleanup sooner rather than later
	delete[] data;

	// Generate 1M random numbers on host
	start_time = std::chrono::high_resolution_clock::now();

	thrust::host_vector<float> host_vec(SIZE);
	thrust::generate(host_vec.begin(), host_vec.end(), RandomFloatGenerator());

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	printf("Thrust Time to generate data: %lu ms %lu us\n", dt_ms.count(), dt_us.count());


	// Transfer to device
	start_time = std::chrono::high_resolution_clock::now();

	thrust::device_vector<float> device_vec = host_vec;

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	printf("Thrust data transfer time: %lu ms %lu us\n", dt_ms.count(), dt_us.count());

	// Find median with thrust
	start_time       = std::chrono::high_resolution_clock::now();
	float gpu_median = thrust_find_median(device_vec);
	end_time         = std::chrono::high_resolution_clock::now();
	dt_ms            = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us            = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("Thrust Median %f\nThrust Time to find median: %lu ms %lu us\n", gpu_median, dt_ms.count(), dt_us.count());

	return 0;
}
