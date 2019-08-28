#pragma once
#include <filesystem>
#include <string>
#include "concurrentqueue.h"
#include "blockingconcurrentqueue.h"
#include "../util/pixels.h"


/**
 * \brief Represents the state an image can be in while going through the pipeline.
 * TODO: Remove the now unused states.
 */
enum image_state
{
	/**
	 * \brief Normally not used, only when not initialized.
	 */
	uninitialized = 1 << 0,
	/**
	 * \brief We have "found" the image on disk, i.e. initialized.
	 */
	found = 1 << 1,
	/**
	 * \brief We are done reading in the image and its EXIF metadata.
	 */
	read_done = 1 << 2,
	/**
	 * \brief The image was fully decompressed.
	 */
	decompressed = 1 << 3,
	/**
	 * \brief We are completely done with this image and it can now be discarded.
	 */
	done = 1 << 4
};

/**
 * \brief Represents an input image that is sent down the pipeline.
 */
struct image
{
	/**
	 * \brief The chronological id, as determined by ManagerWorker. The first input image gets id 0, the second id 1, and so on.
	 */
	long long id = -1;
	/**
	 * \brief The state of the image, \link image_state \endlink
	 */
	image_state state = uninitialized;
	/**
	 * \brief The path of the image.
	 */
	std::string path;
	/**
	 * \brief The time the image was taken as determined by the EXIF Metadata.
	 */
	time_t timestamp = -1;
	/**
	 * \brief The jpeg data as read in from the file.
	 */
	char* jpeg_data = nullptr;
	/**
	 * \brief The size of the jpeg data (i.e. the file).
	 */
	long long jpeg_size = -1;
	long width = 0;
	long height = 0;
	/**
	 * \brief The raw data of the image after being decoded from the jpeg.
	 * Now unused.
	 */
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

/**
 * \brief Parse the time read from the EXIF metadata of an image.
 * \param time The time string as it was read from the EXIF metadata.
 * \return Timestamp from epoch.
 */
time_t parse_time(std::string time);

void compress_image(rgb_pixel* raw, int width, int height, char*& output, unsigned long& output_size);

/**
 * \brief Convert a state into a string. \link image_state \endlink
 * \param state 
 * \return 
 */
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
