#include "DispatchWorker.h"
#include "PerformanceCounter.h"

PERF_COUNTER_INIT(dispatch)

output_frame* DispatchWorker::Process(image* current)
{
	output_frame* output = this->manager->GetForImage(current);

	if (output != nullptr)
	{
		// Check if we already have all the necessary frames, otherwise don't bother checking if they are all decompressed.
		// Also check if we haven't already started the rendering process!
		if (output->inputs.size() == output->expected_count && output->state == created)
		{
			bool all_good = true;
			for (auto input : output->inputs)
			{
				if (input->state != decompressed)
				{
					all_good = false;
					break;
				}
			}

			if (all_good)
			{
				//this->logger->debug("All inputs to this output are fine! output_id: {}", output->id);
				output->state = dispatched;
				PERF_COUNT_STEP(dispatch);
				return output;
				//this->output->enqueue(output);
			}
		}
	}

	return nullptr;
}
