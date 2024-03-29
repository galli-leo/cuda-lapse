#pragma once
#include "BaseWorker.h"
#include "../items/image.h"

class DirectoryWorker :
	public BaseWorker<void*, image*>
{
public:
	using BaseWorker::BaseWorker;

	void Run() override;

	string directory;

	string name() override { return "DirectoryWorker"; }

private:
	int current_id = 0;
};

