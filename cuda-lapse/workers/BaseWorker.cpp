#include "BaseWorker.h"
#include "../items/image.h"
#include "../items/output_frame.h"


template <class TInput, class TOutput>
BaseWorker<TInput, TOutput>::BaseWorker(moodycamel::BlockingConcurrentQueue<TInput>* input, moodycamel::BlockingConcurrentQueue<TOutput>* output) : input(input), output(output)
{
	this->logger = create_logger(this->name());
}

template <class TInput, class TOutput>
BaseWorker<TInput, TOutput>::~BaseWorker()
{
}

template <class TInput, class TOutput>
void BaseWorker<TInput, TOutput>::Start()
{
	thread = std::thread(&BaseWorker::Run, this);
}

template <class TInput, class TOutput>
void BaseWorker<TInput, TOutput>::Join()
{
	thread.join();
}

template <class TInput, class TOutput>
void BaseWorker<TInput, TOutput>::Run()
{
	auto this_id = std::hash<std::thread::id>{}(std::this_thread::get_id());

	this->logger->info("Starting thread: {}", this_id);
	while(!this->done)
	{
		TInput current;

		this->input->wait_dequeue(current);

		// Check if current is null and thus we are done
		if (current == nullptr)
		{
			this->Stop();

			this->logger->info("Stopping thread: {}", this_id);

			if (this->output != nullptr)
			{
				// Signal to next stage that we are done.
				this->output->enqueue(nullptr);
			}
			// Signal to other threads on this stage that they should also stop
			this->input->enqueue(nullptr);

			break;
		} else
		{
			// Otherwise process item and append to output queue.
			TOutput output = this->Process(current);
			// Make sure we have an output queue and our output is not null (otherwise, we would tell the next stages to cancel)
			if (this->output != nullptr && output != nullptr)
			{
				this->output->enqueue(output);
			}
		}
	}
}

template <class TInput, class TOutput>
TOutput BaseWorker<TInput, TOutput>::Process(TInput current)
{
	return nullptr;
}

template <class TInput, class TOutput>
void BaseWorker<TInput, TOutput>::Stop()
{
	this->done = true;
}

/*
 * Instantiate these templates so the linker does not freak out :)
 */
template class BaseWorker < image*, image*>;
template class BaseWorker <void*, image*>;
template class BaseWorker <image*, void*>;
template class BaseWorker <image*, output_frame*>;
template class BaseWorker <output_frame*, output_frame*>;
template class BaseWorker <output_frame*, void*>;