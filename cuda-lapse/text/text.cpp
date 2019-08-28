#include "text.h"
#include <algorithm>

/**
 * \brief Zero frame, used to initialize an empty text.
 */
static frame zero = frame{ 0, 0, 0, 0 };

text create_text(std::string str, atlas* font)
{
	text ret;
	ret.frame = zero;

	int x = 0;
	unsigned int height = 0;
	for (char& c : str)
	{
		if (c != ' ')
		{
			auto txt_char = create_text_char(c, font);
			txt_char.frame.pos.x = x;
			x += txt_char.frame.size.width;

			ret.characters.push_back(txt_char);
			height = std::max(height, txt_char.frame.size.height);
		}
		else
		{
			x += font->space_size;
		}
	}

	ret.frame.size.width = x;
	ret.frame.size.height = height;

	return ret;
}

void move_text(text* txt, int dx, int dy)
{
	for (auto& old_char : txt->characters)
	{
		old_char.frame.pos.x += dx;
		old_char.frame.pos.y += dy;
	}
}

void add_text(text* txt, text* new_text)
{
	move_text(new_text, 0, txt->frame.size.height); // Move new_text down by the height of old text.
	txt->frame.size.height += new_text->frame.size.height; // Expand current text by new_text height.

	int width_diff = txt->frame.size.width - new_text->frame.size.width;
	int x_shift = abs(width_diff) / 2;

	if (width_diff < 0.0) // Check if we need to expand width.
	{
		// If we need to expand width, x coordinates of new_text chars are fine, but we need to shift the x coordinates of all the chars in txt so that it'size still centered!
		/*    txt before			txt after
		 *    | a b c |			|    a b c    | coords here have been shifted!
		 * | a b c d e f | =>	| a b c d e f |
		 *	 new_text
		 */

		move_text(txt, x_shift, 0);
		txt->frame.size.width = new_text->frame.size.width; // We only need to expand the width of the old text, new text can stay as it was.
	}
	else
	{
		// We need to shift our new text since it'size smaller than the old text
		/*    txt before			txt after
		 * | a b c d e f |		| a b c d e f |
		 *    | a b c |		 =>	|    a b c    | coords here have been shifted!
		 *	  new_text
		 */
		move_text(new_text, x_shift, 0);
	}

	txt->characters.insert(txt->characters.end(), new_text->characters.begin(), new_text->characters.end());
}

text_char create_text_char(char c, atlas* font)
{
	text_char txt_char;
	txt_char.character = font->characters[c];
	txt_char.frame.pos.x = 0;
	txt_char.frame.pos.y = 0;
	txt_char.frame.size = txt_char.character.frame.size;
	txt_char.font = font;

	return txt_char;
}

void add_line(text* txt, std::string str, atlas* font)
{
	text new_text = create_text(str, font);

	add_text(txt, &new_text);
}
