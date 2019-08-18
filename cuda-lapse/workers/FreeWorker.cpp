#include "FreeWorker.h"

void* FreeWorker::Process(image* current)
{
	free(current->raw_data);
	current->raw_data = nullptr;

	return nullptr;
}
