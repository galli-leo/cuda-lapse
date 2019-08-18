#pragma once
#include "BaseWorker.h"
#include "concurrentqueue.h"
#include "../images.h"
#include "../logger.h"

class EXIFWorker :
	public BaseWorker<image*, image*>
{
public:
	using BaseWorker::BaseWorker;

	image* Process(image* current) override;

	string name() override { return "EXIFWorker"; }

private:
};

