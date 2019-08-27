#pragma once
#include "../images.h"
#include <vector>
#include <cuda_runtime_api.h>


void blend_frames(std::vector<image> frames, rgba_pixel* &result);

void blend_directly(rgb_pixel **frames, rgba_pixel *output, int count, int max, int blocks, int threads_per_block);

void blend_single(rgb_pixel *frame, rgba_pixel *output, int max, int blocks, int threads_per_block, cudaStream_t stream);