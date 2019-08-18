#include <iostream>
#include <filesystem>
#include "cli.h"
#include "clipp.h"
#include "cuda/cuda_util.h"

program_configuration config;

bool parse_args(int argc, char* argv[])
{
	using namespace clipp;

	char* exe_file = argv[0];
	filesystem::path exe_path = filesystem::path(exe_file);
	config.exe_directory = exe_path.parent_path().string();

	// Parsed Arguments
	bool show_help = false;
	bool show_version = false;
	bool show_devices = false;

	auto cli = (
		option("-v", "--version").set(show_version).doc("Show version and exit."),
		option("-h", "--help").set(show_help).doc("Show this help text."),
		option("-l", "--list-devices").set(show_devices).doc("Shows available cuda devices as well as available encoding params for each device."),
		// Speedup Options
		option("-d", "--day").doc("Specify the speedup to use during working hours [10].") & value("speedup", config.day_speedup),
		option("-n", "--night").doc("Specify the speedup to use during non-working hours (e.g. at night and on the weekend) [20].") & value("speedup", config.night_speedup),
		// Threading
		option("-t", "--threads").doc("Specify the number of CPU threads to use for reading in images.") & value("num", config.threads),
		option("-r", "--delay").doc("Specify the amount of delay for the reader worker per file.") & value("delay", config.reader_delay),
		// Output file
		option("-o", "--output").doc("Output video file. This should end in a h264 supported container!") & value("output", config.output),
		// Directory
		value("directory", config.directory).doc("Directory containing the time lapse images.")
		);

	const auto parse_result = parse(argc, argv, cli);

	bool show_info = show_help || show_version || show_devices;

	if (parse_result && !show_info)
	{
		return true;
	}

	if (!show_info)
	{
		cout << "Invalid arguments! Usage:\n";
	}

	if (show_version)
	{
		cout << PROGNAME << " " << VERSION << "\n";
	}
	else if(show_devices)
	{
		list_cuda_devices(true);
	}else
	{
		cout << make_man_page(cli, PROGNAME);
	}

	return false;
}

void initialize_params(NV_ENC_INITIALIZE_PARAMS& params)
{
	params.frameRateNum = config.fps;
}