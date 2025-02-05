#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

const size_t SIZE = 1e6; // 1,000,000



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


float *cpu_add_vec(float *arr1, float *arr2, size_t size1, size_t size2)
{
	if(size1 != size2)
	{
		throw std::runtime_error("Tried to cpu add vec when sizes are not the same!");
	}

	float *output_arr = new float[size1];

	for(size_t i = 0; i < size1; i++)
	{
		output_arr[i] = arr1[i] + arr2[i];
	}

	return output_arr;
}


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


// CUDA vector addition, this happens before using thrust
__global__ void vector_add_kernel(const float *a, const float *b, float *c, size_t size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
	{
		c[idx] = a[idx] + b[idx];
	}
}

__host__ float thrust_find_median(float *device_data, size_t size)
{
	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(device_data);

	// This sorts the device ptr in place
	thrust::sort(thrust::device, dev_ptr, dev_ptr + size);

	size_t midpoint = size / 2;
	float  mid1     = dev_ptr[midpoint];

	if(size & 0b01 == 0)
	{
		float mid2 = dev_ptr[midpoint - 1];
		return (mid1 + mid2) / 2.0f;
	}

	return mid1;
}



int main()
{
	srand(time(0));

	// Time data creation
	std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

	float *data1 = generate_array(SIZE);
	float *data2 = generate_array(SIZE);

	std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();

	std::chrono::milliseconds dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	std::chrono::microseconds dt_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("CPU Time to generate data: %lu ms %lu us\n", dt_ms.count(), dt_us.count());


	// Time CPU add
	start_time = std::chrono::high_resolution_clock::now();

	float *summed_vecs = cpu_add_vec(data1, data2, SIZE, SIZE);

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("CPU add time: %lu ms %lu us\n", dt_ms.count(), dt_us.count());


	// Time finding median
	start_time = std::chrono::high_resolution_clock::now();

	float median = cpu_median(summed_vecs, SIZE);

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("CPU Median: %f\nTime to find CPU median: %lu ms %lu us\n", median, dt_ms.count(), dt_us.count());

	// Cleanup sooner rather than later
	delete[] data1;
	delete[] data2;
	delete[] summed_vecs;

	// Generate 1M random numbers on host
	start_time = std::chrono::high_resolution_clock::now();

	float *data_a;
	float *data_b;
	float *data_c; // output
	cudaMalloc(&data_a, SIZE * sizeof(float));
	cudaMalloc(&data_b, SIZE * sizeof(float));
	cudaMalloc(&data_c, SIZE * sizeof(float));

	thrust::host_vector<float> host_vec_a(SIZE);
	thrust::host_vector<float> host_vec_b(SIZE);

	thrust::generate(host_vec_a.begin(), host_vec_a.end(), RandomFloatGenerator());
	thrust::generate(host_vec_b.begin(), host_vec_b.end(), RandomFloatGenerator());


	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	printf("Thrust Time to generate data: %lu ms %lu us\n", dt_ms.count(), dt_us.count());


	// Transfer to device
	start_time = std::chrono::high_resolution_clock::now();

	cudaMemcpy(data_a, host_vec_a.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(data_b, host_vec_b.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("CUDA data transfer time (for generated data, coming from CPU generation): %lu ms %lu us\n", dt_ms.count(), dt_us.count());

	// Time vector add kernel


	start_time = std::chrono::high_resolution_clock::now();

	int threadcount = 256;
	int blocks      = (SIZE + threadcount - 1) / threadcount;
	vector_add_kernel<<<blocks, threadcount>>>(data_a, data_b, data_c, SIZE);
	cudaDeviceSynchronize();

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("CUDA vector add kernel time: %lu ms %lu ms\n", dt_ms.count(), dt_us.count());

	// Find median with thrust
	start_time = std::chrono::high_resolution_clock::now();

	float gpu_median = thrust_find_median(data_c, SIZE);

	end_time = std::chrono::high_resolution_clock::now();
	dt_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	dt_us    = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

	printf("Thrust Median %f\nThrust Time to find median: %lu ms %lu us\n", gpu_median, dt_ms.count(), dt_us.count());

	return 0;
}
