#include "cuda_util.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <nvcuvid.h>
#include <nvEncodeAPI.h>
#include <string>
#include <vector>
#include <map>
#include "../nvcuvid/Encoder/NvEncoderCuda.h"
#include "../nvcuvid/NvCodecUtils.h"

using namespace std;

const static string tab = "\t";
const static string middle = "|-- ";
const static string bottom = "\\-- ";

template <typename T>
void print_table(const string& label, T value, int level = 1, bool is_bottom = false)
{
	for (int i = 0; i < level; i++)
	{
		cout << tab;
	}

	cout << (is_bottom ? bottom : middle);
	cout << label << ": " << value << endl;
}

void list_encoders(int device)
{
	CUcontext context;

	CUdevice dev;
	cuDeviceGet(&dev, device);

	CUresult result = cuCtxCreate(&context, 0, dev);

	ck(result);

	auto encoder = NvEncoderCuda(context, 1280, 720, NV_ENC_BUFFER_FORMAT_ARGB);

	const static map<string, GUID> formats = { {"h264", NV_ENC_CODEC_H264_GUID}, {"HEVC", NV_ENC_CODEC_HEVC_GUID} };

	const static map<string, NV_ENC_CAPS> caps = { {"Rate Controls Modes", NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES}, {"Width Max", NV_ENC_CAPS_WIDTH_MAX}, {"Height Max", NV_ENC_CAPS_HEIGHT_MAX} };

	for (auto const&[name, guid] : formats)
	{
		print_table(name + " Encoder", "", 2, true);

		for (auto const&[cap_name, cap_key] : caps)
		{
			int value = encoder.GetCapabilityValue(guid, cap_key);
			print_table(cap_name, value, 3, false);
		}
	}

	encoder.DestroyEncoder();
	cuCtxDestroy(context);
}

void list_cuda_devices(bool include_encoders)
{
	int nDevices;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		cout << "Device Number: " << i << endl;
		print_table("Device Name", prop.name);

		print_table("Memory Configuration", "", 1, true);
		print_table("Memory Clock Rate (KHz)", prop.memoryClockRate, 2, false);
		print_table("Memory Bus Width (bits)", prop.memoryBusWidth, 2, false);
		print_table("Peak Memory Bandwidth (GB/s)", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6, 2, true);

		print_table("Thread Configuration:", "", 1, true);
		print_table("Threads per Block", prop.maxThreadsPerBlock, 2);
		print_table("Threads per Processor", prop.maxThreadsPerMultiProcessor, 2);
		print_table("Num Processor", prop.multiProcessorCount, 2, true);

		if (include_encoders)
		{
			print_table("Encoders", "", 1, true);
			list_encoders(i);
		}
	}
}

std::unordered_map<int, size_t> devices_and_memory()
{
	int nDevices;

	std::unordered_map<int, size_t> ret;

	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaSetDevice(i);
		//cudaDeviceProp prop;
		//cudaGetDeviceProperties(&prop, i);
		size_t free;
		size_t total;
		cudaMemGetInfo(&free, &total);
		ret[i] = free;
	}

	return ret;
}

