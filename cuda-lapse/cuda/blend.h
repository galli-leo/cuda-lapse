#pragma once
#include <vector>
#include <cuda_runtime_api.h>

#include "../util/pixels.h"
#include "../text/cuda_text.h"

void blend_directly(rgb_pixel **frames, rgba_pixel *output, int count, int max, int blocks, int threads_per_block);

void blend_single(rgb_pixel *frame, rgba_pixel *output, int max, int count, int blocks, int threads_per_block, cudaStream_t stream);

void render_text(cuda_text text, rgba_pixel* output, int max, int width, int blocks, int threads_per_block);