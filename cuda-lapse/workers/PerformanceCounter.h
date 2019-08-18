#pragma once
#include <atomic>
#include <string>
#include <spdlog/logger.h>

class PerformanceCounter
{
public:
	PerformanceCounter(std::string name, long long milliseconds = 1000 * 5);
	~PerformanceCounter();

	void Step(long long step = 1);

private:
	std::atomic<long long> count;
	long long prev_count = -1;
	std::atomic<long long> prev_time;

	long long milliseconds;
	std::string name;

	std::shared_ptr<spdlog::logger> logger;

	void Report(long long count, long long cur_time, long long prev_time);
};

#define PERF_COUNTER_INIT(name) static PerformanceCounter* counter_##name = new PerformanceCounter(#name);

#define PERF_COUNT_STEP(name) counter_##name->Step();
