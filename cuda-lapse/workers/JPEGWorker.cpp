#include "JPEGWorker.h"
#include "../images_fmt.h"
#include "PerformanceCounter.h"

PERF_COUNTER_INIT(jpeg)

image* JPEGWorker::Process(image* current)
{
	int width = 4000;
	int height = 3000;

	current->raw_size = width * height * sizeof(rgb_pixel);

	current->raw_data = static_cast<rgb_pixel*>(malloc(current->raw_size));

	tjDecompress2(this->decompressor, reinterpret_cast<unsigned char*>(current->jpeg_data), current->jpeg_size,
		reinterpret_cast<unsigned char*>(current->raw_data), width, 0, height, TJPF_RGB,
		TJFLAG_FASTDCT);

	free(current->jpeg_data);
	current->jpeg_data = nullptr;
	current->state = decompressed;

	PERF_COUNT_STEP(jpeg)

	return current;
}
