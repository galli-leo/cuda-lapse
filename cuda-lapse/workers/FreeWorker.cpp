#include "FreeWorker.h"

void* FreeWorker::Process(image* current)
{
	free(current->jpeg_data);
	current->jpeg_data = nullptr;

	return nullptr;
}
