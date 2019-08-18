#include "EXIFWorker.h"
#include <iostream>
#include <fstream>
#include "TinyEXIF.h"
#include "../images_fmt.h"
#include "PerformanceCounter.h"
#include "../cli.h"
#include "../ram_usage.h"

PERF_COUNTER_INIT(exif)

image* EXIFWorker::Process(image* current)
{
	ifstream file(current->path, ifstream::binary | ifstream::in);

	current->jpeg_data = static_cast<char *>(malloc(current->jpeg_size));

	try
	{
		file.read(current->jpeg_data, current->jpeg_size);

		//logger->debug("Parsing EXIF info");

		TinyEXIF::EXIFInfo image_exif((uint8_t *)current->jpeg_data, current->jpeg_size);
		if (image_exif.Fields)
		{
			auto time = parse_time(image_exif.DateTime);
			current->timestamp = time;
			current->width = image_exif.ImageWidth;
			current->height = image_exif.ImageHeight;
			//logger->debug("Parsed EXIF info: Time: {}, WxH: {}x{}, Parsed Time: {}", image_exif.DateTime, image_exif.ImageWidth, image_exif.ImageHeight, time);
		}
		else
		{
			logger->error("Error parsing exif data for image: {}!", *current);
		}

		if (current->state == found)
		{
			current->state = read_done;
		}

		PERF_COUNT_STEP(exif)

			// Give other workers a chance to work
			//std::this_thread::sleep_for(std::chrono::milliseconds(config.reader_delay));

			//int output_size = this->output->size_approx();

			size_t ram_usage = getCurrentRSS();
		size_t max_ram = 1024;
		max_ram = max_ram * max_ram * max_ram * 20;

		//int delay = config.reader_delay;
		int times = 1;

		while (ram_usage > max_ram)
		{
			this->logger->debug("Currently using more than 20 GB of RAM, slowing down: {}", ram_usage);
			// Give other workers a chance to work
			std::this_thread::sleep_for(std::chrono::milliseconds(config.reader_delay * times));
			ram_usage = getCurrentRSS();
			if (times <= 5) times++;
		}

	}
	catch (exception& e)
	{
		this->logger->error("Error while trying to read file: {}, error: {}", current->path, e.what());
	}

	return current;
}
