
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "blend.h"
#include <stdio.h>

using namespace std;

__global__ void blendKernel(rgb_pixel **frames, rgba_pixel *output, int count, int max)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < max)
	{
		output[i].alpha = 0xff;
		output[i].red = 0;
		output[i].green = 0;
		output[i].blue = 0;
		for (int j = 0; j < count; j++)
		{
			output[i].red += 1 / 10.0 * frames[j][i].red;
			output[i].green += 1 / 10.0 * frames[j][i].green;
			output[i].blue += 1 / 10.0 * frames[j][i].blue;
		}
	}
}

__global__ void blendSingleKernel(rgb_pixel *frame, rgba_pixel *output, int max)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < max)
	{
		unsigned int red = frame[i].red * 1 / 10.0;
		unsigned int green = frame[i].green * 1 / 10.0;
		unsigned int blue = frame[i].blue * 1 / 10.0;
		unsigned int value = (0xff & red) + ((green << 8) & 0xff00) + ((blue << 16) & 0xff0000);
		//atomicAdd((int*)&output[i], value);
		unsigned int* output_loc = (unsigned int*)&output[i];
		atomicAdd(output_loc, value);
		//*output_loc = *output_loc + value;
		output[i].alpha = 0xff;
	}
}

#define THREADS_PER_BLOCK 1024

void blend_directly(rgb_pixel **frames, rgba_pixel *output, int count, int max, int blocks, int threads_per_block)
{
	blendKernel << <blocks, threads_per_block >> > (frames, output, count, max);
}

void blend_single(rgb_pixel *frame, rgba_pixel *output, int max, int blocks, int threads_per_block, cudaStream_t stream)
{
	blendSingleKernel << <blocks, threads_per_block >> > (frame, output, max);
}


cudaError_t blendWithCuda(vector<image> frames, int width, int height, rgba_pixel* result)
{
	vector<rgb_pixel*> inputs = vector<rgb_pixel*>();
	rgba_pixel* output = nullptr;
	rgb_pixel** dev_inputs;

	int frame_size = width * height * sizeof(rgb_pixel);
	int output_size = width * height * sizeof(rgba_pixel);

	cudaError_t cudaStatus = cudaSuccess;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate buffers

	cudaStatus = cudaMalloc((void**)&output, output_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	for (auto frame : frames)
	{
		rgb_pixel* tmp = nullptr;
		cudaStatus = cudaMalloc((void**)&tmp, frame_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}
		inputs.push_back(tmp);
	}

	int inputs_size = inputs.size() * sizeof(rgb_pixel*);

	cudaStatus = cudaMalloc((void**)&dev_inputs, inputs_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	int i = 0;
	for (auto frame : frames)
	{
		auto input = inputs.at(i);
		cudaStatus = cudaMemcpy(input, frame.raw_data, frame_size, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}
		i++;
	}

	cudaStatus = cudaMemcpy(dev_inputs, inputs.data(), inputs_size, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!: %s", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	int blocks = (width * height) / THREADS_PER_BLOCK;

	printf("Launching kernel with %d blocks and %d threads per block (%dx%d)\n", blocks, THREADS_PER_BLOCK, width, height);

	blendKernel << <blocks, THREADS_PER_BLOCK>> > (dev_inputs, output, 10, width * height);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "blendKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result, output, output_size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	// Free resources
	cudaFree(output);

	for (auto input : inputs)
	{
		cudaFree(input);
	}

	return cudaStatus;
}

void blend_frames(vector<image> frames, rgba_pixel*& result)
{
	auto first = frames.at(0);
	int width = first.width;
	int height = first.height;

	result = static_cast<rgba_pixel*>(malloc(width * height * sizeof(rgba_pixel)));

	blendWithCuda(frames, width, height, result);
}