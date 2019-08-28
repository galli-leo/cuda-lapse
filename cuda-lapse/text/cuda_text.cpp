#include "cuda_text.h"

cuda_text_char convert_char_to_cuda(text_char txt_char, int device)
{
	cuda_text_char ret;
	ret.frame = txt_char.frame;
	ret.atlas_frame = txt_char.character.frame;
	ret.cuda_atlas = txt_char.font->cuda_png_data.at(device);

	return ret;
}
