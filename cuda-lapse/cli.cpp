#include <iostream>
#include <filesystem>

#include "cli.h"
#pragma warning(push, 0)
#include "clipp.h"
#pragma warning(pop)
#include "cuda/cuda_util.h"
#include "util/pixels.h"

program_configuration config;

long picture_size;
long input_size;
long output_size;

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
		// Input
		option("-pw", "--width").doc("Specify the width of the input image.") & value("width", config.picture_width),
		option("-ph", "--height").doc("Specify the height of the input image.") & value("height", config.picture_height),
		// Output file
		option("-o", "--output").doc("Output video file. This should end in a h264 supported container!") & value("output", config.output),
		// Fonts
		option("-df", "--date-font").doc("Specify the path to the json of the font used for displaying the date.") & value("font-64.json", config.date_font),
		option("-tf", "--time-font").doc("Specify the path to the json of the font used for displaying the time.") & value("font-80.json", config.time_font),
		// Directory
		value("directory", config.directory).doc("Directory containing the time lapse images.")
		);

	const auto parse_result = parse(argc, argv, cli);

	bool show_info = show_help || show_version || show_devices;

	if (parse_result && !show_info)
	{
		picture_size = config.picture_width * config.picture_height;
		input_size = picture_size * sizeof(rgb_pixel);
		output_size = picture_size * sizeof(rgba_pixel);
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
