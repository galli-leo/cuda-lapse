#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "../cuda-lapse/cuda/blend.h";
#include "../cuda-lapse/lodepng/lodepng.h";

int main(int argc, char* argv[])
{
	if (argc < 3)
	{
		std::cout << "Error to few arguments!" << std::endl;
		return 1;
	}
	
	std::string font_name = std::string(argv[2]);

	std::string image_name = std::string(argv[1]);

	std::cout << "Loading font" << std::endl;

	atlas font = read_atlas(font_name);
	
	rgba_pixel* output;
	
	size_t width = 4000;
	size_t height = 3000;
	
	size_t picture_size = width * height;
	
	size_t output_size = picture_size * sizeof(rgba_pixel);

	cudaMalloc(&output, output_size);

	text txt = create_text("15:28", &font);
	add_line(&txt, "Tue, 26. Aug", &font);
	txt.frame.pos.x = 0;
	txt.frame.pos.y = 0;

	cuda_text_char* cuda_chars;

	int size_chars = sizeof(cuda_text_char) * txt.characters.size();

	cudaMalloc(&cuda_chars, size_chars);

	std::vector<cuda_text_char> host_cuda_chars;
	for (auto txt_char : txt.characters)
	{
		host_cuda_chars.push_back(convert_char_to_cuda(txt_char, 0));
	}

	cudaMemcpy(cuda_chars, host_cuda_chars.data(), size_chars, cudaMemcpyHostToDevice);

	cuda_text cuda_txt{
		cuda_chars,
		txt.characters.size(),
		txt.frame
	};

#define THREADS_PER_BLOCK 1024

	int blocks = (picture_size) / THREADS_PER_BLOCK;
	
	render_text(cuda_txt, output, picture_size, 4000, blocks, THREADS_PER_BLOCK);

	cudaDeviceSynchronize();

	rgba_pixel* host_output = static_cast<rgba_pixel*>(malloc(output_size));

	cudaMemcpy(host_output, output, output_size, cudaMemcpyDeviceToHost);

	lodepng_encode32_file("output.png", reinterpret_cast<unsigned char*>(host_output), width, height);

	// cudaDeviceReset must be called before exiting in order for profiling and
// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	
    return 0;
}
