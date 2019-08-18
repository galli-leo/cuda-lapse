#pragma once
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/sink.h"
#include "spdlog/common.h"

#define LOG_INIT(name)	auto static logger = create_logger(name);

using namespace std;

inline shared_ptr<spdlog::sinks::sink> create_file_sink(spdlog::level::level_enum level, int num_files = 3, long max_size = 1024 * 1024 * 5)
{
	auto file_sink = make_shared<spdlog::sinks::rotating_file_sink_mt> (string(spdlog::level::to_short_c_str(level)) + "_log.txt", max_size, num_files);
	file_sink->set_level(level);
	return file_sink;
}

inline shared_ptr<spdlog::logger> create_logger(string module)
{
	auto console_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
	console_sink->set_level(spdlog::level::trace);

	auto debug_file_sink = create_file_sink(spdlog::level::debug);
	auto info_file_sink = create_file_sink(spdlog::level::info);
	auto warning_file_sink = create_file_sink(spdlog::level::warn);

	spdlog::logger logger(module, { console_sink, debug_file_sink, info_file_sink, warning_file_sink });

	logger.set_level(spdlog::level::trace);

	return make_shared<spdlog::logger>(logger);
}