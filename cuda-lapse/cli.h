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
	int picture_width = 4000;
	int picture_height = 3000;
	string output = "output";
	string directory;
	string exe_directory;
	string date_font;
	string time_font;
};

extern program_configuration config;

/*
 * Precompute sizes needed for the space of storing input / output frames.
 */
static long picture_size;
static long input_size;
static long output_size;

/**
 * \brief Parses arguments in to the global config. Returns false if parsing failed / help is shown (indicates exit).
 * \param argc number of arguments in argv
 * \param argv array of arguments
 * \return 
 */
bool parse_args(int argc, char* argv[]);

void initialize_params(NV_ENC_INITIALIZE_PARAMS& params);