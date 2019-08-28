#include "ManagerWorker.h"
#include "../items/image_fmt.h"

#define MIN_SPEEDUP 10

void ManagerWorker::CreateOutputFrame(image* current)
{
	output_frame* output = new output_frame;
	output->id = this->current_output_id;
	output->timestamp = current->timestamp;
	output->start_id = this->next_input_id;
	output->expected_count = InputFramesForTimestamp(output->timestamp);
	output->end_id = this->next_input_id + output->expected_count;
	output->state = created;
	output->width = current->width;
	output->height = current->height;
	output->inputs.push_back(current);

	for (long long id = output->start_id; id < output->end_id; id++)
	{
		this->inputs_to_output[id] = output;
		// Check if we have any pending inputs that match this output frame
		image* pending = this->pending_inputs[id];
		if (pending != nullptr)
		{
			output->inputs.push_back(pending);
			// Remove the pending input, since we now handled it!
			this->pending_inputs.erase(id);
		}
	}

	// Increment output id and set correct next_input_id
	this->current_output_id++;
	this->next_input_id = output->end_id;
}

long long ManagerWorker::InputFramesForTimestamp(time_t time)
{
	// TODO: Do an actual calculation here!
	return MIN_SPEEDUP;
}

image* ManagerWorker::Process(image* current)
{
	// Check if this image could be the data source for our next output frame
	if (this->next_input_id <= current->id && current->id < this->next_input_id + MIN_SPEEDUP)
	{
		this->CreateOutputFrame(current);

		// Now we check if we have any pending inputs with ids in the correct range and create output frames for them too.

		bool found = false;
		do
		{
			found = false;
			// Go through all possible images that could be in the next output frame, until we find one.
			for (long long id = this->next_input_id; id < this->next_input_id + MIN_SPEEDUP; id++)
			{
				image* pending = this->pending_inputs[id];

				if (pending != nullptr)
				{
					// We found a pending image! Create a new output frame and then start searching again.
					found = true;
					this->CreateOutputFrame(pending);
					break;
				}
			}

		} while (found);
	}
	else if (current->id < this->next_input_id)
	{
		// This means we already created the corresponding output_frame. Search that and add this image.
		output_frame* output = this->GetForImage(current);
		output->inputs.push_back(current);
	}
	else
	{
		// This means we haven't yet decided anything for the previous output frame(s) and hence we need to wait with this image.
		this->pending_inputs[current->id] = current;
	}

	return current;
}

output_frame* ManagerWorker::GetForImage(image* image)
{
	return this->GetForID(image->id);
}

output_frame* ManagerWorker::GetForID(long long id)
{
	if (inputs_to_output.find(id) == inputs_to_output.end())
		return nullptr;

	return inputs_to_output[id];
}
