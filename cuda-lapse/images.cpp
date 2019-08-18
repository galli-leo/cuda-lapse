#include <set>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>

#include <turbojpeg.h>

#include "images.h"
#include "util.h"
#include "TinyEXIF.h"
#include "logger.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"

const static set<string> image_exts = { ".png", ".jpg", ".jpeg"};

const static tjhandle jComp = tjInitCompress();

LOG_INIT("images")

bool is_image(filesystem::directory_entry entry)
{
	if (entry.is_directory()) return false;

	const auto path = entry.path();

	if (!path.has_extension()) return false;

	auto extension = path.extension().string();

	lowercased(extension);

	return image_exts.find(extension) != image_exts.end();
}

time_t parse_time(string time)
{
	std::tm tm{};
	istringstream iss;
	iss.str(time);
	iss >> std::get_time(&tm, "%Y:%m:%d %H:%M:%S");
	return mktime(&tm);
}

void compress_image(rgb_pixel* raw, int width, int height, char*& output, unsigned long& output_size)
{
	const int QUALITY = 100;

	int result = tjCompress2(jComp, reinterpret_cast<unsigned char*>(raw), width, 0, height, TJPF_RGB, reinterpret_cast<unsigned char**>(&output), &output_size, TJSAMP_444, QUALITY, TJFLAG_FASTDCT);

	if (result < 0)
	{
		logger->error("Error while trying to compress image: {}", tjGetErrorStr2(jComp));
	}

	logger->debug("Compressed output size: {}", output_size);
}





