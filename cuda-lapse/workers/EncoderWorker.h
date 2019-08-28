#pragma once
#include "BaseWorker.h"
#include "../items/image.h"
#include "../items/output_frame.h"
#include "../nvcuvid/MP4Encoder.h"

class EncoderWorker :
	public BaseWorker<output_frame*, void*>
{
public:
	using BaseWorker::BaseWorker;

	void InitializeEncoder(string output_file);

	void* Process(output_frame* current) override;

	/**
	 * \brief Render the output frame.
	 * \param frame 
	 */
	void RenderFrame(output_frame* frame);

	/**
	 * \brief Finish up the encoding.
	 */
	void FinishEncoding();

	string output_file;

	string name() override { return "EncoderWorker"; }

private:

	int current_frame = 0;

	vector<output_frame*> pending_frames;

	MP4Encoder* encoder;
};

