#pragma once
#include <filesystem>
#include <string>
#include "concurrentqueue.h"
#include "blockingconcurrentqueue.h"


enum image_state
{
	uninitialized = 1 << 0,
	found = 1 << 1,
	read_done = 1 << 2,
	decompressed = 1 << 3,
	done = 1 << 4
};

struct rgb_pixel
{
	unsigned char red;
	unsigned char green;
	unsigned char blue;
};

struct rgba_pixel
{
	unsigned char red;
	unsigned char green;
	unsigned char blue;
	unsigned char alpha;
};

struct image
{
	long long id = -1;
	image_state state = uninitialized;
	std::string path;
	time_t timestamp = -1;
	char* jpeg_data = nullptr;
	long long jpeg_size = -1;
	long width = 0;
	long height = 0;
	rgb_pixel* raw_data = nullptr;
	long long raw_size = -1;


};

/*
 * Queue typedefs
 */

typedef moodycamel::ConcurrentQueue<image*> ImageQueue;
typedef moodycamel::BlockingConcurrentQueue<image*> BlockingImageQueue;

/**
 * \brief Returns whether the given directory entry is an image file based on the file extension. (Allowed extensions are .jpg, .jpeg, .png).
 * \param entry 
 * \return 
 */
#ifndef __CUDACC__
bool is_image(std::filesystem::directory_entry entry);
#endif

time_t parse_time(std::string time);

void compress_image(rgb_pixel* raw, int width, int height, char*& output, unsigned long& output_size);

inline std::string state_to_string(image_state state)
{
	switch (state)
	{
	case uninitialized:
		return "uninitialized";
	case read_done:
		return "read_done";
	case found:
		return "found";
	case decompressed:
		return "decompressed";
	case done:
		return "done";

	}
	return "error";
}
