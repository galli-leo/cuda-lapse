#pragma once

#include <fstream>
#include <string>
#include "Encoder/NvEncoderCuda.h"
#include "../util/logger.h"

struct rgba_pixel;
using namespace std;

class MP4Encoder
{
public:
	MP4Encoder(string output_file, int width = 4000, int height = 3000);
	~MP4Encoder();

	void EncodeFrame(rgba_pixel* frame);
	void FinishEncode();
	void WritePacketToFile(std::vector<std::vector<uint8_t>>& vPacket);
	int GetNumberFrames();

private:
	shared_ptr<spdlog::logger> logger = create_logger("MP4Encoder");
	string output_file;
	string mp4_file;
	ofstream output_stream;

	CUcontext context;
	NvEncoderCuda* encoder;
	unsigned int num_frames_encoded = 0;
};

