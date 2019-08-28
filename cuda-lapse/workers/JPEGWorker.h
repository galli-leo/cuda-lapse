#pragma once
#include <turbojpeg.h>

#include "BaseWorker.h"
#include "../items/image.h"

class JPEGWorker :
	public BaseWorker<image*, image*>
{
public:
	using BaseWorker::BaseWorker;

	image* Process(image* current) override;

	string name() override { return "JPEGWorker"; }

private:

	tjhandle decompressor = tjInitDecompress();
};

