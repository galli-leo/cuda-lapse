#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <set>
#include "cli.h"
#include "clipp.h"
#include "images.h"
#include "cuda/cuda_util.h"
#include "cuda/blend.h"
#include <turbojpeg.h>
#include <thread>
#include "logger.h"

#include "nvcuvid/Encoder/NvEncoderCuda.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "MP4Encoder.h"
#include "workers/BaseWorker.h"
#include "workers/DirectoryWorker.h"
#include "workers/EXIFWorker.h"
#include "workers/JPEGWorker.h"
#include "workers/RenderWorker.h"
#include "workers/EncoderWorker.h"
#include "workers/ManagerWorker.h"
#include "workers/DispatchWorker.h"
#include "workers/FreeWorker.h"

using namespace std;

LOG_INIT("main")

int main(int argc, char* argv[])
{
	if (parse_args(argc, argv))
	{
		logger->info("Exe is in directory: {}, using this many threads: {}", config.exe_directory, config.threads);

		logger->debug("CUDA devices:");

		list_cuda_devices();

		// Image Queues
		BlockingImageQueue* dirToEXIF = new BlockingImageQueue();
		BlockingImageQueue* exifToManager = new BlockingImageQueue();
		BlockingImageQueue* managerToJPEG = new BlockingImageQueue();
		BlockingImageQueue* jpegToDispatcher = new BlockingImageQueue();
		BlockingImageQueue* renderToFree = new BlockingImageQueue();

		// Output Queues
		BlockingOutputQueue* dispatcherToRender = new BlockingOutputQueue();
		BlockingOutputQueue* renderToEncoder = new BlockingOutputQueue();

		vector<BaseWorkerBase*> workers;

		// Create directory worker
		DirectoryWorker* dirWorker = BaseWorker<void*, image*>::CreateAndStart<DirectoryWorker>(workers, nullptr, dirToEXIF);
		dirWorker->directory = config.directory;

		// Create EXIF workers
		BaseWorker<image*, image*>::CreateAndStartMany<EXIFWorker>(workers, dirToEXIF, exifToManager, 1);

		// Create Manager worker
		ManagerWorker* managerWorker = BaseWorker<image*, image*>::CreateAndStart<ManagerWorker>(workers, exifToManager, managerToJPEG);

		// Create JPEG workers
		BaseWorker<image*, image*>::CreateAndStartMany<JPEGWorker>(workers, managerToJPEG, jpegToDispatcher, config.threads);

		// Create Dispatcher worker
		DispatchWorker* dispatchWorker = BaseWorker<image*, output_frame*>::CreateAndStart<DispatchWorker>(workers, jpegToDispatcher, dispatcherToRender);
		dispatchWorker->manager = managerWorker;
		//dispatchWorker->Start();

		auto vram_map = devices_and_memory();

		const long needed_vram = RenderWorker::VRAMNeeded();
		RenderWorker* testWorker = nullptr;
		// Create Render workers
		for (auto pair : vram_map)
		{
			int device = pair.first;
			size_t vram = pair.second;

			int threads = vram / needed_vram / 0.6;

			logger->info("Creating {} Render Threads for Device {} with available VRAM {}", threads, device, vram);

			BaseWorker<output_frame*, output_frame*>::CreateAndStartMany<RenderWorker>(workers, dispatcherToRender, renderToEncoder, threads, [device, renderToFree, &testWorker](RenderWorker* renderWorker)
			{
				renderWorker->device = device;
				renderWorker->free = renderToFree;
				testWorker = renderWorker;
			});
		}

		BaseWorker<image*, void*>::CreateAndStartMany<FreeWorker>(workers, renderToFree, nullptr, 10);
		//create_and_start_many<RenderWorker, BlockingOutputQueue*, BlockingOutputQueue*>(workers, dispatcherToRender, renderToEncoder, 10);

		// Create Encoder worker
		EncoderWorker* encoderWorker = BaseWorker<output_frame*, void*>::CreateAndStart<EncoderWorker>(workers, renderToEncoder, nullptr);
		encoderWorker->InitializeEncoder(config.output);
		//create_and_start<EncoderWorker, BlockingOutputQueue*, string>(workers, renderToEncoder, config.output);


		for (auto worker : workers)
		{
			worker->Start();
		}

		logger->info("Waiting on all workers to finish working!");

		for (auto worker : workers)
		{
			worker->Join();
		}

		logger->info("All workers finished!");

		encoderWorker->FinishEncoding();
	}

	return 0;
}