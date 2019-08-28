#pragma once
#include "BaseWorker.h"
#include "../items/image.h"

class FreeWorker :
	public BaseWorker<image*, void*>
{
public:
	using BaseWorker::BaseWorker;

	void* Process(image* current) override;

	string name() override { return "FreeWorker"; }
};

