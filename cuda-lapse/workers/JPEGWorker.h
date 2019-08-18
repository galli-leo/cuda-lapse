#pragma once
#include "BaseWorker.h"
#include "../images.h"
#include "../logger.h"
#include <turbojpeg.h>

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

