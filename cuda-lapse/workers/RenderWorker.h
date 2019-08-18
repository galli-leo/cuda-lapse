#pragma once
#include "BaseWorker.h"
#include "../images.h"
#include "../logger.h"
#include "../output_frame.h"
#include <cuda_runtime.h>

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
	static long VRAMNeeded(int width, int height);

	BlockingImageQueue* free;

	int device = 0;

	string name() override { return "RenderWorker"; }

private:

	rgba_pixel* dev_output;
	vector<rgb_pixel*> dev_inputs;
	rgb_pixel** dev_inputs_arr;

	bool LogError(string message, cudaError_t error);
};

