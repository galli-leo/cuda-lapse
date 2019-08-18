#pragma once
#include "BaseWorker.h"
#include "../images.h"
#include "../output_frame.h"
#include "ManagerWorker.h"

class DispatchWorker :
	public BaseWorker<image*, output_frame*>
{
public:
	using BaseWorker::BaseWorker;

	output_frame* Process(image* current) override;

	ManagerWorker* manager;

	string name() override { return "DispatchWorker"; }
};

