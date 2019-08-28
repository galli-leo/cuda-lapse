#pragma once
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include "BaseWorker.h"
#include "../util/pixels.h"
#include "../items/image.h"
#include "../items/output_frame.h"
#include "../text/atlas.h"


class RenderWorker :
	public BaseWorker<output_frame*, output_frame*>
{
public:
	using BaseWorker::BaseWorker;

	output_frame* Process(output_frame* current) override;

	void Stop() override;

	void Run() override;

	/**
	 * \brief Initialize the device used for cuda as well as prepare all the memory.
	 * \param device 
	 */
	void InitializeDevice(int device = 0);

	/**
	 * \brief Returns the amount of VRAM needed to process images with width x height.
	 * \param width 
	 * \param height 
	 * \return 
	 */
	static long VRAMNeeded();

	BlockingImageQueue* free;

	int device = 0;

	string name() override { return "RenderWorker"; }

	atlas time_font;
	atlas date_font;

	string time_font_name;
	string date_font_name;

private:

	rgba_pixel* dev_output;
	vector<rgb_pixel*> dev_inputs;
	rgb_pixel** dev_inputs_arr;

	vector<cudaStream_t> streams;

	nvjpegHandle_t handle;
	nvjpegJpegState_t jpeg_state;

	template<typename... Args>
	bool LogError(cudaError_t error, spdlog::source_loc loc, string message, const Args &... args);

	bool LogJPEGError(string message, nvjpegStatus_t error);
};

