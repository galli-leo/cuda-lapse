#include "RenderWorker.h"
#include "../cuda/blend.h"
#include "PerformanceCounter.h"
#include "../cli.h"
#include <nvjpeg.h>

PERF_COUNTER_INIT(render)

#define THREADS_PER_BLOCK 1024

output_frame* RenderWorker::Process(output_frame* current)
{
	this->LogError("cudaMalloc Failed for output", cudaMalloc(&dev_output, output_size));

	int blocks = (picture_size) / THREADS_PER_BLOCK;

	int i = 0;
	for (auto input : current->inputs)
	{
		cudaStream_t stream = this->streams.at(i);

		rgb_pixel* tmp = nullptr;
		this->LogError("cudaMalloc failed for input", cudaMalloc(&tmp, input_size));
		this->dev_inputs.push_back(tmp);

		nvjpegImage_t image;
		image.channel[0] = reinterpret_cast<unsigned char*>(tmp);
		image.pitch[0] = config.picture_width * 3;

		this->LogJPEGError("failed to decode input", nvjpegDecode(this->handle, this->jpeg_state, reinterpret_cast<unsigned char*>(input->jpeg_data), input->jpeg_size, NVJPEG_OUTPUT_RGBI, &image, stream));

		blend_single(tmp, this->dev_output, picture_size, blocks, THREADS_PER_BLOCK, stream);

		i++;
	}

	for (auto stream : this->streams)
	{
		cudaStreamSynchronize(stream);
	}

	current->output = static_cast<rgba_pixel*>(malloc(current->width * current->height * sizeof(rgba_pixel)));

	this->LogError("cudaMemcpy failed while trying to copy back resulting frame", cudaMemcpy(current->output, this->dev_output, output_size, cudaMemcpyDeviceToHost));

	current->state = rendered;

	// Free decompressed data

	for (auto dev_input : this->dev_inputs)
	{
		cudaFree(dev_input);
	}

	cudaFree(this->dev_output);

	this->dev_inputs.clear();

	for (auto input : current->inputs)
	{
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

	for (int i = 0; i < MAX_INPUTS; i++)
	{
		cudaStream_t tmp;
		this->LogError("cudaStreamCreate failed", cudaStreamCreate(&tmp));
		this->streams.push_back(tmp);
	}

	int inputs_arr_size = sizeof(rgb_pixel*)*MAX_INPUTS;//this->dev_inputs.size();

	this->LogError("cudaMalloc failed for input arr", cudaMalloc(&dev_inputs_arr, inputs_arr_size));

	this->LogJPEGError("createSimple failed", nvjpegCreate(NVJPEG_BACKEND_GPU_HYBRID, NULL, &this->handle));

	this->LogJPEGError("Creating state failed", nvjpegJpegStateCreate(this->handle, &this->jpeg_state));

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

static unordered_map<nvjpegStatus_t, tuple<string, string>> jpeg_status_to_info = {
	{NVJPEG_STATUS_SUCCESS, {"SUCCESS", "The API call has finished successfully. Note that many of the calls are asynchronous and some of the errors may be seen only after synchronization."}},
	{NVJPEG_STATUS_NOT_INITIALIZED, {"NOT_INITIALIZED", "The library handle was not initialized. A call to nvjpegCreate() is required to initialize the handle."}},
	{NVJPEG_STATUS_INVALID_PARAMETER, {"INVALID_PARAMETER", "Wrong parameter was passed. For example, a null pointer as input data, or an image index not in the allowed range."}},
	{NVJPEG_STATUS_BAD_JPEG,{"BAD_JPEG", "Cannot parse the JPEG stream. Check that the encoded JPEG stream and its size parameters are correct."}},
	{NVJPEG_STATUS_JPEG_NOT_SUPPORTED,{"JPEG_NOT_SUPPORTED", "Attempting to decode a JPEG stream that is not supported by the nvJPEG library."}},
	{NVJPEG_STATUS_ALLOCATOR_FAILURE,{"ALLOCATOR_FAILURE", "The user-provided allocator functions, for either memory allocation or for releasing the memory, returned a non-zero code."}},
	{NVJPEG_STATUS_EXECUTION_FAILED,{"EXECUTION_FAILED", "Error during the execution of the device tasks."}},
	{NVJPEG_STATUS_ARCH_MISMATCH,{"ARCH_MISMATCH", "The device capabilities are not enough for the set of input parameters provided (input parameters such as backend, encoded stream parameters, output format)."}},
	{NVJPEG_STATUS_INTERNAL_ERROR,{"INTERNAL_ERROR", "Error during the execution of the device tasks."}},
	{NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED,{"IMPLEMENTATION_NOT_SUPPORTED", "Not supported."}},
};

bool RenderWorker::LogJPEGError(string message, nvjpegStatus_t error)
{
	if (error == NVJPEG_STATUS_SUCCESS) return true;

	tuple<string, string> info = jpeg_status_to_info[error];

	this->logger->error("{}: [{}]", message, get<0>(info), get<1>(info));

	return false;
}
