#include "RenderWorker.h"
#include "../cuda/blend.h"
#include "PerformanceCounter.h"
#include "../cli.h"
#pragma warning(push, 0)        
#include "spdlog/fmt/fmt.h"
#pragma warning(pop)

#include <nvjpeg.h>
#include "../util/date_names.h"

PERF_COUNTER_INIT(render)

#define THREADS_PER_BLOCK 1024

#define CHECK(status, message, ...) this->LogError(status, spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, message, __VA_ARGS__)

output_frame* RenderWorker::Process(output_frame* current)
{
	CHECK(cudaMalloc(&dev_output, output_size), "cudaMalloc failed to allocate output buffer.");

	int blocks = (picture_size) / THREADS_PER_BLOCK;

	int i = 0;
	for (auto input : current->inputs)
	{
		cudaStream_t stream = this->streams.at(i);

		rgb_pixel* tmp = nullptr;
		CHECK(cudaMalloc(&tmp, input_size), "cudaMalloc failed for input");
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

	auto lt = localtime(&current->timestamp);

	// To make it more visually interesting we select a random int to be our minute in the range 0 to 10;
	int min = lt->tm_min - (lt->tm_min % 10);
	min += current->id % 10;

	string time_str = fmt::format("{:02}:{:02}", lt->tm_hour, min);
	string date_str = fmt::format("{}, {:02}. {}", short_weekday(lt->tm_wday), lt->tm_mday, short_month(lt->tm_mon));

	text txt = create_text(time_str, &this->time_font);
	add_line(&txt, date_str, &this->date_font);
	txt.frame.pos.x = 3000;
	txt.frame.pos.y = 2500;
	anchor(&txt.frame, 0.5, 0.5);

	cuda_text_char* cuda_chars;
	
	int size_chars = sizeof(cuda_text_char) * txt.characters.size();

	CHECK(cudaMalloc(&cuda_chars, size_chars), "cudaMalloc Failed for text characters");

	vector<cuda_text_char> host_cuda_chars;
	for (auto txt_char : txt.characters)
	{
		host_cuda_chars.push_back(convert_char_to_cuda(txt_char, this->device));
	}

	CHECK(cudaMemcpy(cuda_chars, host_cuda_chars.data(), size_chars, cudaMemcpyHostToDevice), "cudaMemcpy failed while trying to copy text chars");

	cuda_text cuda_txt{
		cuda_chars,
		txt.characters.size(),
		txt.frame
	};

	render_text(cuda_txt, this->dev_output, picture_size, 4000, blocks, THREADS_PER_BLOCK);

	CHECK(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed after launching text render kernel");

	current->output = static_cast<rgba_pixel*>(malloc(current->width * current->height * sizeof(rgba_pixel)));

	CHECK(cudaMemcpy(current->output, this->dev_output, output_size, cudaMemcpyDeviceToHost), "cudaMemcpy failed while trying to copy back resulting frame");

	current->state = rendered;

	// Free decompressed data

	for (auto dev_input : this->dev_inputs)
	{
		cudaFree(dev_input);
	}

	cudaFree(this->dev_output);

	cudaFree(cuda_chars);

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

#if !DEBUG_CUDA
	cudaSetDevice(device);
#endif
	// Allocate buffers

	for (int i = 0; i < MAX_INPUTS; i++)
	{
		cudaStream_t tmp;
		CHECK(cudaStreamCreate(&tmp), "cudaStreamCreate failed");
		this->streams.push_back(tmp);
	}

	int inputs_arr_size = sizeof(rgb_pixel*)*MAX_INPUTS;//this->dev_inputs.size();

	CHECK(cudaMalloc(&dev_inputs_arr, inputs_arr_size), "cudaMalloc failed for input arr");

	this->LogJPEGError("createSimple failed", nvjpegCreate(NVJPEG_BACKEND_GPU_HYBRID, NULL, &this->handle));

	this->LogJPEGError("Creating state failed", nvjpegJpegStateCreate(this->handle, &this->jpeg_state));

	/*this->time_font = read_atlas(this->time_font_name);
	this->date_font = read_atlas(this->date_font_name);*/
	
	this->logger->debug("Finished allocating device memory");
}

long RenderWorker::VRAMNeeded()
{
	return input_size * MAX_INPUTS + 2*output_size;
}

template<typename... Args>
bool RenderWorker::LogError(cudaError_t error, spdlog::source_loc loc, string message, const Args& ... args)
{
	if (error == cudaSuccess) return true;
	this->logger->log(loc, spdlog::level::err, "{}: (#{}) [{}] {}", fmt::format(message, args...), this->device, cudaGetErrorName(error), cudaGetErrorString(error));

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
