#pragma once
#define NOMINMAX
#include <string>
#include <nvEncodeAPI.h>
using namespace std;

#define VERSION "v0.1 supra alpha"
#define PROGNAME "cuda-lapse"

struct program_configuration
{
	int day_speedup = 10;
	int night_speedup = 20;
	int fps = 24;
	int threads = 10;
	int reader_delay = 200;
	string output = "output";
	string directory;
	string exe_directory;
};

extern program_configuration config;

/**
 * \brief Parses arguments in to the global config. Returns false if parsing failed / help is shown (indicates exit).
 * \param argc number of arguments in argv
 * \param argv array of arguments
 * \return 
 */
bool parse_args(int argc, char* argv[]);

void initialize_params(NV_ENC_INITIALIZE_PARAMS& params);