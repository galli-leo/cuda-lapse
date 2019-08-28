#include "DirectoryWorker.h"

#include "PerformanceCounter.h"

PERF_COUNTER_INIT(directory)

void DirectoryWorker::Run()
{
	logger->info("Scanning directory {} for image files.", this->directory);
	int counter = 0;
	for (auto& p : filesystem::recursive_directory_iterator(this->directory))
	{
		if (is_image(p))
		{
			// Create new image object that is sent down the pipeline.
			image* current = new image;
			current->state = uninitialized;
			current->width = 4000;
			current->height = 3000;
			current->path = p.path().string();
			current->jpeg_size = p.file_size();
			current->id = this->current_id;

			this->current_id++;

			PERF_COUNT_STEP(directory);

			this->output->enqueue(current);

			counter++;

			//if (counter == 1000) break;
		}
	}

	this->output->enqueue(nullptr);
}
