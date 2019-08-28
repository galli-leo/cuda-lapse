#pragma once
#include <vector>

#include "../util/geometry.h"
#include "atlas.h"

/**
 * \brief Represents a single character inside a text object to be rendered.
 * Contains the character as seen on the font atlas, as well as the frame of the character relative to the frame of the whole text object.
 */
struct text_char
{
	character character;
	frame frame;
	/**
	 * \brief Reference to the font we want to use to render this character.
	 * This is needed for CUDA.
	 */
	atlas* font;
};

/**
 * \brief Represents text to be rendered.
 * Can span multiple lines.
 * The positioning of characters is controlled entirely by the frame of each character itself (relative to the whole text object).
 */
struct text
{
	/**
	 * \brief The characters that we want to render.
	 */
	std::vector<text_char> characters;
	/**
	 * \brief The frame of the whole text, characters are rendered relative to this.
	 */
	frame frame;
};

/**
 * \brief Create a text struct from the given string and font atlas.
 * \param str The string to create a text from.
 * \param font The font atlas to use.
 * \return Ready to use text struct.
 */
text create_text(std::string str, atlas* font);

/**
 * \brief Move all of the characters of the given text by (\p dx, \p dy).
 * This is mostly used to add more characters to existing text.
 * \param txt The text to move.
 * \param dx Amount of movement in the x direction.
 * \param dy Amount of movement in the y direction.
 */
void move_text(text* txt, int dx, int dy);

/**
 * \brief Add the characters from \p new_text to \p txt.
 * This function currently adds the characters on the line below the existing characters.
 * It also takes care of expanding the frame if necessary and makes sure all text is still centered.
 * \param txt The text which we want to append characters to.
 * \param new_text The text containing the characters we want to append.
 */
void add_text(text* txt, text* new_text);

/**
 * \brief Create a text_char based upon \p c and \p font.
 * Takes care of initializing the text_char frame.
 * \param c
 * \param font
 * \return
 */
text_char create_text_char(char c, atlas* font);

/**
 * \brief Add a new line of text to an existing text struct.
 * Basically just a call to \link create_text \endlink and \link add_text \endlink.
 * \param txt
 * \param str
 * \param font
 */
void add_line(text* txt, std::string str, atlas* font);