#pragma once
#include <thread>
#include "../util/logger.h"
#include "blockingconcurrentqueue.h"

using namespace std;

class BaseWorkerBase
{
public:
	virtual void Start() = 0;

	virtual void Join() = 0;
};

template <class TInput, class TOutput>
class BaseWorker : BaseWorkerBase
{
public:
	typedef moodycamel::BlockingConcurrentQueue<TInput> InputQueue;
	typedef moodycamel::BlockingConcurrentQueue<TOutput> OutputQueue;

	BaseWorker(InputQueue* input, OutputQueue* output);
	~BaseWorker();

	/**
	 * \brief Start this worker on a separate thread.
	 */
	void Start() override;

	/**
	 * \brief Wait on this worker to finish.
	 */
	void Join() override;

	/**
	 * \brief Code that is run on this thread.
	 */
	virtual void Run();

	/**
	 * \brief Process one item and "convert" it into output so it can be processed by next stage.
	 * \param current Current item to be processed.
	 * \return 
	 */
	virtual TOutput Process(TInput current);

	/**
	 * \brief Stops this worker from doing any more work.
	 */
	virtual void Stop();

	/**
	 * \brief Create a Worker of type T and add it to the workers array.
	 * \tparam T The worker type.
	 * \param workers Array where the resulting worker should be stored.
	 * \param input The input queue.
	 * \param output The output queue.
	 * \return Pointer to the newly created worker.
	 */
	template <class T>
	static T* CreateAndStart(vector<BaseWorkerBase*>& workers, InputQueue* input, OutputQueue* output)
	{
		T* worker = new T(input, output);
		workers.push_back(worker);
		worker->logger = create_logger(worker->name());
		return worker;
	}




	/**
	 * \brief Create count many workers of type T and add them to the workers array. Also run init for each worker.
	 * \tparam T 
	 * \param workers 
	 * \param output 
	 * \param count 
	 * \param init 
	 */
	template <class T>
	static void CreateAndStartMany(vector<BaseWorkerBase*>& workers, InputQueue* input, OutputQueue* output,
		int count, std::function<void(T*)> const& init = {})
	{
		for (int i = 0; i < count; i++)
		{
			T* worker = CreateAndStart<T>(workers, input, output);
			if (init) init(worker);
		}
	}

	bool done = false;

	shared_ptr<spdlog::logger> logger;

protected:
	virtual string name() { return "BaseWorker"; };

	InputQueue* input;
	OutputQueue* output;


private:
	thread thread;
};

