#include "EncoderWorker.h"
#include <utility>
#include "PerformanceCounter.h"

PERF_COUNTER_INIT(encoder)

void EncoderWorker::InitializeEncoder(string output_file)
{
	this->output_file = output_file;
	this->encoder = new MP4Encoder(output_file);
}

void* EncoderWorker::Process(output_frame* current)
{
	// Check whether this is the frame we currently want to render
	if (current->id == current_frame)
	{
		this->RenderFrame(current);
		// Iterate over all pending frames in order of id and check whether one of them is something we want to render now.
		vector<output_frame*> rendered;
		sort(this->pending_frames.begin(), this->pending_frames.end(), [](output_frame* const a, output_frame* const b)
		{
			return a->id < b->id;
		});
		for (auto frame : this->pending_frames)
		{
			if (frame->id == current_frame)
			{
				this->RenderFrame(frame);
				rendered.push_back(frame);
			}
		}

		// Remove rendered frames from pending
		for (auto frame : rendered)
		{
			this->pending_frames.erase(std::remove(this->pending_frames.begin(), this->pending_frames.end(), frame), this->pending_frames.end());
		}
	}
	else
	{
		// Otherwise add it to the pending frames list
		this->pending_frames.push_back(current);
		//this->logger->debug("Received non current frame: {}, we want to render: {}", current->id, current_frame);
		if (this->pending_frames.size() > 5)
		{
			stringstream ss;
			for (auto frame : pending_frames)
			{
				ss << frame->id << ", ";
			}

			//this->logger->info("Pending frames: {}", ss.str());
		}
	}

	return nullptr;
}

void EncoderWorker::RenderFrame(output_frame* frame)
{
	this->encoder->EncodeFrame(frame->output);
	// Free frame output;
	free(frame->output);
	frame->output = nullptr;
	this->current_frame = frame->id + 1;

	PERF_COUNT_STEP(encoder)
}

void EncoderWorker::FinishEncoding()
{
	this->encoder->FinishEncode();
}
