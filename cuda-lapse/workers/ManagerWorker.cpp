#include "ManagerWorker.h"
#include "../items/image_fmt.h"
#include "../cli.h"

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

	if (output->id % 40 == 0)
	{
		this->logger->debug("Calculated expected_count: {}, for time: {}", output->expected_count, asctime(localtime(&output->timestamp)));
	}

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

int sec_of_day(const tm* tm)
{
	return tm->tm_hour * 60 * 60 + tm->tm_min * 60 + tm->tm_sec;
}

int sec_of_week(const tm* tm)
{
	return tm->tm_wday * 24 * 60 * 60 + sec_of_day(tm);
}

const int morning_hour = 7;
const int night_hour = 19;

const tm night_to_day = tm{ 0, 0, morning_hour };
const tm day_to_night = tm{ 0, 0, night_hour };

const tm friday_to_saturday = tm{ 0, 0, night_hour, 0, 0, 0, 5 };
const tm sunday_to_monday = tm{ 0, 0, morning_hour, 0, 0, 0, 1 };

const int morning = sec_of_day(&night_to_day);
const int night = sec_of_day(&day_to_night);
const int friday = sec_of_week(&friday_to_saturday);
const int monday = sec_of_week(&sunday_to_monday);

const double euler = std::exp(1.0);

int ManagerWorker::InputFramesForTimestamp(time_t time)
{
	auto lt = localtime(&time);

	int day_secs = sec_of_day(lt);
	int week_secs = sec_of_week(lt);

	int almost_full = 100 * 60; // In seconds

	double L = 1, k = 0.02 / (almost_full / 100.0);
	double x0;
	int d = config.day_speedup;
	int c = (config.night_speedup - config.day_speedup);
	int kmod = 1;
	int x; // Input

	int morning_diff = morning - day_secs;
	int night_diff = night - day_secs;

	int friday_diff = friday - week_secs;
	int monday_diff = monday - week_secs;

	if (friday_diff <= 0 || monday_diff >= 0)
	{
		// This input frame was taken on the weekend!
		x = week_secs;
		if (abs(friday_diff) < abs(monday_diff))
		{
			// Closer to friday, than monday.
			x0 = friday;
		} else
		{
			x0 = monday;
			kmod = -1;
		}
	} else
	{
		// Not on weekend!
		x = day_secs;
		if (abs(morning_diff) < abs(night_diff))
		{
			// Closer to morning
			x0 = morning;
			kmod = -1;
		} else
		{
			x0 = night;
		}
	}

	k = kmod * k;
	
	// Logistic Function
	double output = L / (1 + pow(euler, -k * (x - x0)));
	
	return static_cast<int>(round(c * output + d));
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
