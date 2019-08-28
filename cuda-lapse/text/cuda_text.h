#pragma once
#include "../util/geometry.h"
#include "../util/pixels.h"
#include "text.h"

struct cuda_text_char
{
	frame atlas_frame;
	frame frame;
	rgba_pixel* cuda_atlas;
};

struct cuda_text
{
	cuda_text_char* characters;
	unsigned int num_characters;
	frame frame;
};

/**
 * \brief Converts a text_char to a cuda_text_char we use for rendering.
 * \param txt_char 
 * \param device 
 * \return 
 */
cuda_text_char convert_char_to_cuda(text_char txt_char, int device = 0);