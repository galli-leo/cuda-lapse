#include "JPEGWorker.h"
#include "../items/image_fmt.h"
#include "PerformanceCounter.h"
#include "../cli.h"

PERF_COUNTER_INIT(jpeg)

image* JPEGWorker::Process(image* current)
{
	return current;
	current->raw_size = input_size;

	current->raw_data = static_cast<rgb_pixel*>(malloc(current->raw_size));

	tjDecompress2(this->decompressor, reinterpret_cast<unsigned char*>(current->jpeg_data), current->jpeg_size,
		reinterpret_cast<unsigned char*>(current->raw_data), config.picture_width, 0, config.picture_height, TJPF_RGB,
		TJFLAG_FASTDCT);

	free(current->jpeg_data);
	current->jpeg_data = nullptr;
	current->state = decompressed;

	PERF_COUNT_STEP(jpeg)

	return current;
}
