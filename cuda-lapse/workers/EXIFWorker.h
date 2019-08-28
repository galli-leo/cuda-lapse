#pragma once
#include "BaseWorker.h"
#include "../items/image.h"

class EXIFWorker :
	public BaseWorker<image*, image*>
{
public:
	using BaseWorker::BaseWorker;

	image* Process(image* current) override;

	string name() override { return "EXIFWorker"; }

private:
};

