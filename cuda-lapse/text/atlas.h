#pragma once
#include <vector>
#include <string>
#include <unordered_map>

#include "../util/geometry.h"
#include "../util/pixels.h"

/**
 * \brief Represents a character present on the font atlas texture.
 */
struct character
{
	/**
	 * \brief The actual character represented by this struct.
	 */
	char c;
	/**
	 * \brief The rectangle occupied on the font atlas texture.
	 */
	frame frame;
};

/**
 * \brief Represents a font atlas texture.
 */
struct atlas
{
	/**
	 * \brief Mapping of the characters found in this font atlas to the frames on the texture.
	 */
	std::unordered_map<char, character> characters;
	/**
	 * \brief The filename used to load the png file.
	 */
	std::string png_file;
	/**
	 * \brief The data present inside the png file.
	 */
	unsigned char* png_data;
	/**
	 * \brief Contains a pointer to the decoded png data for every cuda device.
	 */
	std::vector<rgba_pixel*> cuda_png_data;
	/**
	 * \brief The size of the png file.
	 */
	frame_size png_size;
	/**
	 * \brief The size of a space character. Currently this is set to the width of the 'f' character.
	 */
	size_t space_size;
};

/**
 * \brief Read an atlas found at \p font_name.
 * Note that \p font_name should have the extension removed.
 * We will automatically append .json to load the character map and .png to load the png file.
 *
 * Important: This function will automatically allocate memory on every connected cuda device for the font texture.
 * It will also automatically copy over the decoded png data.
 * \param font_name 
 * \return 
 */
atlas read_atlas(std::string font_name);