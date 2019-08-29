#include <string>
#include <filesystem>

#include "cli.h"
#include "clipp.h"
#include "items/image.h"
#include "cuda/cuda_util.h"
#include "cuda/blend.h"
#include "util/logger.h"
#include "nvcuvid/Encoder/NvEncoderCuda.h"
#include "workers/workers.h"

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
		BaseWorker<image*, image*>::CreateAndStartMany<EXIFWorker>(workers, dirToEXIF, exifToManager, 2);

		// Create Manager worker
		ManagerWorker* managerWorker = BaseWorker<image*, image*>::CreateAndStart<ManagerWorker>(workers, exifToManager, managerToJPEG);

		// Create JPEG workers
		BaseWorker<image*, image*>::CreateAndStartMany<JPEGWorker>(workers, managerToJPEG, jpegToDispatcher, config.threads);

		// Create Dispatcher worker
		DispatchWorker* dispatchWorker = BaseWorker<image*, output_frame*>::CreateAndStart<DispatchWorker>(workers, jpegToDispatcher, dispatcherToRender);
		dispatchWorker->manager = managerWorker;
		//dispatchWorker->Start();

		atlas time_font = read_atlas(config.time_font);
		atlas date_font = read_atlas(config.date_font);

		// We don't want multiple threads if we are trying to debug our cuda kernel. We also do not want to execute cudaSetDevice anytime, otherwise debugging cuda code just straight up doesn't work :(
#if !DEBUG_CUDA
		auto vram_map = devices_and_memory();
		
		const long needed_vram = RenderWorker::VRAMNeeded();
		// Create Render workers
		for (auto pair : vram_map)
		{
			int device = pair.first;
			size_t vram = pair.second;
#else
			int device = 0;
#endif
#if DEBUG_CUDA
			int threads = 4;
#else
			int threads = static_cast<int>(vram / needed_vram);

			logger->info("Creating {} Render Threads for Device {} with available VRAM {}", threads, device, vram);
#endif

			BaseWorker<output_frame*, output_frame*>::CreateAndStartMany<RenderWorker>(workers, dispatcherToRender, renderToEncoder, threads, [device, renderToFree, time_font, date_font](RenderWorker* renderWorker)
			{
				renderWorker->device = device;
				renderWorker->free = renderToFree;
				renderWorker->time_font = time_font;//config.time_font;
				renderWorker->date_font = date_font;//config.date_font;
			});
#if !DEBUG_CUDA
		}
#endif

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