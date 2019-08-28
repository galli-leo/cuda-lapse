#include <string>
#include <cuda_runtime.h>
#include <fstream>

#include "atlas.h"
#include "../util/logger.h"
#pragma warning(push, 0)    
#include "../json.hpp"
#pragma warning(pop)
#include "../lodepng/lodepng.h"

auto logger = create_logger("Atlas");

atlas read_atlas(std::string font_name)
{
	auto font_json = font_name + ".json";
	auto font_png = font_name + ".png";

	logger->info("Reading font json file: {}", font_json);

	atlas font;

	font.png_file = font_png;

	std::ifstream json_file(font_json);
	nlohmann::json j;
	json_file >> j;

	auto root = j.get<nlohmann::json::object_t>();

	for (auto& item : root) {
		auto key = item.first;
		auto value = item.second;
		char c = key.at(0);

		character atlas_char;
		atlas_char.c = c;

		atlas_char.frame.pos.x = value["x"];
		atlas_char.frame.pos.y = value["y"];
		atlas_char.frame.size.height = value["height"];
		atlas_char.frame.size.width = value["width"];

		font.characters[c] = atlas_char;
	}

	font.space_size = font.characters['f'].frame.size.width;

	logger->info("Reading font png file: {}", font.png_file);

	lodepng_decode32_file(&font.png_data, &font.png_size.width, &font.png_size.height, font.png_file.c_str());

	logger->info("Loading font atlas image into memory of all cuda devices.");

#if CUDA_TESTING || DEBUG_CUDA
	int numCudaDevices = 1;
#else
	int numCudaDevices = -1;
	cudaGetDeviceCount(&numCudaDevices);
#endif
	for (int i = 0; i < numCudaDevices; i++)
	{
#if !CUDA_TESTING && !DEBUG_CUDA
		cudaSetDevice(i);
#endif
		//logger->info("Loading font atlas image into memory of cuda device #{}", i);

		rgba_pixel* tmp;
		size_t png_data_size = sizeof(rgba_pixel) * font.png_size.height * font.png_size.width;
		cudaMalloc(&tmp, png_data_size);

		cudaMemcpy(tmp, font.png_data, png_data_size, cudaMemcpyHostToDevice);

		font.cuda_png_data.push_back(tmp);
	}

	return font;
}
