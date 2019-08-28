
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "blend.h"
#include <stdio.h>

using namespace std;

__global__ void blendKernel(rgb_pixel **frames, rgba_pixel *output, int count, int max)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < max)
	{
		output[i].alpha = 0xff;
		output[i].red = 0;
		output[i].green = 0;
		output[i].blue = 0;
		for (int j = 0; j < count; j++)
		{
			output[i].red += 1 / 10.0 * frames[j][i].red;
			output[i].green += 1 / 10.0 * frames[j][i].green;
			output[i].blue += 1 / 10.0 * frames[j][i].blue;
		}
	}
}

__global__ void blendSingleKernel(rgb_pixel *frame, rgba_pixel *output, int max)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < max)
	{
		unsigned int red = frame[i].red * 1 / 10.0;
		unsigned int green = frame[i].green * 1 / 10.0;
		unsigned int blue = frame[i].blue * 1 / 10.0;
		unsigned int value = (0xff & red) + ((green << 8) & 0xff00) + ((blue << 16) & 0xff0000);
		//atomicAdd((int*)&output[i], value);
		unsigned int* output_loc = (unsigned int*)&output[i];
		atomicAdd(output_loc, value);
		//*output_loc = *output_loc + value;
		output[i].alpha = 0xff;
	}
}

__device__ bool is_inside_cuda(frame frame, point p)
{
	point end;
	end.x = frame.pos.x + frame.size.width;
	end.y = frame.pos.y + frame.size.height;

	return (frame.pos.x <= p.x && p.x < end.x) && (frame.pos.y <= p.y && p.y < end.y);
}

__device__ bool is_inside_cuda(frame frame, int x, int y)
{
	point p;
	p.x = x;
	p.y = y;
	return is_inside_cuda(frame, p);
}

__global__ void renderTextKernel(cuda_text text, rgba_pixel* output, int max, int width)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < max)
	{
		int x = i % width;
		int y = i / width;
		if (is_inside_cuda(text.frame, x, y))
		{
			int relX = x - text.frame.pos.x;
			int relY = y - text.frame.pos.y;
			// We definitely need to render text here!
			for (int k = 0; k < text.num_characters; k++)
			{
				cuda_text_char txt_char = text.characters[k];
				if (is_inside_cuda(txt_char.frame, relX, relY))
				{
					// We need to render this specific character!
					relX -= txt_char.frame.pos.x;
					relY -= txt_char.frame.pos.y;

					int charX = txt_char.atlas_frame.pos.x + relX;
					int charY = txt_char.atlas_frame.pos.y - txt_char.atlas_frame.size.height + relY;

					rgba_pixel charPix = txt_char.cuda_atlas[charX + charY * 1024];
					float text_alpha = charPix.alpha / 255.0;
					float bg_alpha = 1 - text_alpha;
					output[i].red = output[i].red * bg_alpha + 0xff * text_alpha;
					output[i].green = output[i].green * bg_alpha + 0xff * text_alpha;
					output[i].blue = output[i].blue * bg_alpha + 0xff * text_alpha;
				}
			}
		}
	}
}

#define THREADS_PER_BLOCK 1024

void blend_directly(rgb_pixel **frames, rgba_pixel *output, int count, int max, int blocks, int threads_per_block)
{
	blendKernel << <blocks, threads_per_block >> > (frames, output, count, max);
}

void blend_single(rgb_pixel *frame, rgba_pixel *output, int max, int blocks, int threads_per_block, cudaStream_t stream)
{
	blendSingleKernel << <blocks, threads_per_block >> > (frame, output, max);
}

void render_text(cuda_text text, rgba_pixel* output, int max, int width, int blocks, int threads_per_block)
{
	renderTextKernel << <blocks, threads_per_block >> > (text, output, max, width);
}