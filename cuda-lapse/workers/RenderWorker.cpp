#include "RenderWorker.h"
#include "../cuda/blend.h"
#include "PerformanceCounter.h"
#include "../cli.h"

PERF_COUNTER_INIT(render)

#define THREADS_PER_BLOCK 1024

output_frame* RenderWorker::Process(output_frame* current)
{
	int i = 0;
	for (auto input : current->inputs)
	{
		auto dev_input = this->dev_inputs.at(i);
		this->LogError("cudaMemcpy failed while trying to copy over host input frame data", cudaMemcpyAsync(dev_input, input->raw_data, input_size, cudaMemcpyHostToDevice));
		i++;
	}

	int blocks = (picture_size) / THREADS_PER_BLOCK;

	//this->logger->trace("Launching Kernel with {} blocks and {} threads per block", blocks, THREADS_PER_BLOCK);

	blend_directly(this->dev_inputs_arr, this->dev_output, current->inputs.size(), picture_size, blocks, THREADS_PER_BLOCK);

	//this->logger->trace("Sucessfully launched kernel");

	this->LogError("cudaSynchronize failed", cudaDeviceSynchronize());

	current->output = static_cast<rgba_pixel*>(malloc(current->width * current->height * sizeof(rgba_pixel)));

	this->LogError("cudaMemcpy failed while trying to copy back resulting frame", cudaMemcpy(current->output, this->dev_output, output_size, cudaMemcpyDeviceToHost));

	current->state = rendered;

	// Free decompressed data

	for (auto input : current->inputs)
	{
		//free(input->raw_data);
		//input->raw_data = nullptr;
		this->free->enqueue(input);
	}

	PERF_COUNT_STEP(render)

	return current;
}

void RenderWorker::Stop()
{
	BaseWorker::Stop();

	this->free->enqueue(nullptr);
}

void RenderWorker::Run()
{
	this->InitializeDevice(this->device);

	BaseWorker::Run();
}

void RenderWorker::InitializeDevice(int device)
{
	this->device = device;
	this->logger->debug("Allocating device memory for this worker on device: {}.", device);

	cudaSetDevice(device);

	// Allocate buffers

	this->LogError("cudaMalloc Failed for output", cudaMalloc(&dev_output, output_size));

	for (int i = 0; i < MAX_INPUTS; i++)
	{
		rgb_pixel* tmp = nullptr;
		this->LogError("cudaMalloc failed for input", cudaMalloc(&tmp, input_size));
		this->dev_inputs.push_back(tmp);
	}

	int inputs_arr_size = sizeof(rgb_pixel*)*this->dev_inputs.size();

	this->LogError("cudaMalloc failed for input arr", cudaMalloc(&dev_inputs_arr, inputs_arr_size));

	this->LogError("cudaMemcpy failed when trying to copy array of input pointers", cudaMemcpy(this->dev_inputs_arr, this->dev_inputs.data(), inputs_arr_size, cudaMemcpyHostToDevice));

	this->logger->debug("Finished allocating device memory");
}

long RenderWorker::VRAMNeeded()
{
	return input_size * MAX_INPUTS + 2*output_size;
}

bool RenderWorker::LogError(string message, cudaError_t error)
{
	if (error == cudaSuccess) return true;
	this->logger->error("{}: [{}] {}", message, cudaGetErrorName(error), cudaGetErrorString(error));

	return false;
}
