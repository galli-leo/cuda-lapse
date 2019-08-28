#include "PerformanceCounter.h"
#include "../util/logger.h"


PerformanceCounter::PerformanceCounter(std::string name, long long milliseconds) : name(name), milliseconds(milliseconds)
{
	this->logger = create_logger(name + " PERF");
	this->count.store(0);
	this->prev_time.store(0);
}


PerformanceCounter::~PerformanceCounter()
{
}

#pragma optimize("", off)

void PerformanceCounter::Step(long long step)
{
	long long count = this->count.fetch_add(step);

	auto chrono_now = std::chrono::high_resolution_clock::now();
	auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(chrono_now);
	auto epoch_ms = now_ms.time_since_epoch();
	long long cur_time = epoch_ms.count();
	long long prev_time = this->prev_time.load();
	if (cur_time - prev_time > this->milliseconds && this->prev_time.compare_exchange_strong(prev_time, cur_time))
	{
		this->Report(count, cur_time, prev_time);
	}
}

void PerformanceCounter::Report(long long count, long long cur_time, long long prev_time)
{
	long long diff = count - prev_count;

	long long time_diff = cur_time - prev_time;

	double rate = diff / double(time_diff) * 1000.0;

	this->logger->info("Previous Count: {}, Current Count: {}, Rate: {:.2f} it/s", prev_count, count, rate);

	this->prev_count = count;
}

#pragma optimize("", on)