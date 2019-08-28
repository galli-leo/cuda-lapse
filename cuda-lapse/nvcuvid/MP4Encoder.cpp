#include "MP4Encoder.h"
#include "../cli.h"

MP4Encoder::MP4Encoder(string output_file, int width, int height)
{
	int ret = 0;

	this->output_file = output_file + ".h264";
	this->mp4_file = output_file + ".mp4";

	this->output_stream = ofstream(this->output_file, ios::out | ios::binary);

	this->logger->info("Creating CUDA context and CUDA encoder.");

	CUdevice dev;
	cuDeviceGet(&dev, 0);

	CUresult result = cuCtxCreate(&context, 0, dev);

	if (result < 0)
	{
		this->logger->error("Had error trying to allocate CUDA context: {}", result);
	}

	this->encoder = new NvEncoderCuda(this->context, width, height, NV_ENC_BUFFER_FORMAT_ABGR);

	NV_ENC_INITIALIZE_PARAMS initializeParams = NV_ENC_INITIALIZE_PARAMS{ NV_ENC_INITIALIZE_PARAMS_VER };
	NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
	initializeParams.encodeConfig = &encodeConfig;
	this->encoder->CreateDefaultEncoderParams(&initializeParams, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_DEFAULT_GUID);

	initialize_params(initializeParams);

	this->encoder->CreateEncoder(&initializeParams);
}

MP4Encoder::~MP4Encoder()
{
	this->encoder->DestroyEncoder();
}

void MP4Encoder::EncodeFrame(rgba_pixel* frame)
{
	auto encoder_input = this->encoder->GetNextInputFrame();

	//logger->debug("Next Encoder Frame: Fmt: {}, #planes: {}, chromaOffsets: {}, pitch: {}", encoder_input->bufferFormat, encoder_input->numChromaPlanes, encoder_input->chromaOffsets[0], encoder_input->pitch);

	NvEncoderCuda::CopyToDeviceFrame(context, frame, 0, (CUdeviceptr)encoder_input->inputPtr,
		encoder_input->pitch,
		encoder->GetEncodeWidth(),
		encoder->GetEncodeHeight(),
		CU_MEMORYTYPE_HOST,
		encoder_input->bufferFormat,
		encoder_input->chromaOffsets,
		encoder_input->numChromaPlanes);

	// For receiving encoded packets
	std::vector<std::vector<uint8_t>> vPacket;

	encoder->EncodeFrame(vPacket);

	this->WritePacketToFile(vPacket);
}

void MP4Encoder::FinishEncode()
{
	std::vector<std::vector<uint8_t>> vPacket;
	this->encoder->EndEncode(vPacket);
	this->WritePacketToFile(vPacket);

	this->output_stream.flush();
	this->output_stream.close();

	this->logger->info("Finished Encoding video to h264! Running ffmpeg to mux it into an mp4 video...");

	string command = config.exe_directory + "/ffmpeg.exe -y -i " + this->output_file + " -c:v copy " + this->mp4_file;

	this->logger->info("ffmpeg command to run: {}", command);

	system(command.c_str());
}

void MP4Encoder::WritePacketToFile(std::vector<std::vector<uint8_t>>& vPacket)
{
	this->num_frames_encoded += vPacket.size();
	for (std::vector<uint8_t> &packet : vPacket)
	{
		// For each encoded packet
		this->output_stream.write(reinterpret_cast<char*>(packet.data()), packet.size());
	}

	//this->logger->debug("Wrote {} packet(s) to file.", vPacket.size());
}

int MP4Encoder::GetNumberFrames()
{
	return this->num_frames_encoded;
}
