#pragma once
#include "BaseWorker.h"
#include "../items/image.h"
#include "../items/output_frame.h"
#include <unordered_map>

class ManagerWorker :
	public BaseWorker<image*, image*>
{
public:
	using BaseWorker::BaseWorker;
	
	/**
	 * \brief Create the output frame for the image current. This allocates the space necessary in the maps, etc.
	 * \param current 
	 */
	void CreateOutputFrame(image* current);

	/**
	 * \brief Returns the number of input frames we want to blend for a given timestamp.
	 * \param time 
	 * \return 
	 */
	static int InputFramesForTimestamp(time_t time);

	image* Process(image* current) override;

	/**
	 * \brief Get the output_frame corresponding to the given input image. Basically just calls <code>GetForID(image->id);</code>
	 * \param image 
	 * \return 
	 */
	output_frame* GetForImage(image* image);

	/**
	 * \brief Get the output_frame corresponding to the given input image id.
	 * \param id 
	 * \return 
	 */
	output_frame* GetForID(long long id);

	string name() override { return "ManagerWorker"; }

private:
	long long current_output_id = 0;
	long long current_input_id = -1;
	long long next_input_id = 0;

	unordered_map<long long, output_frame*> inputs_to_output;

	unordered_map<long long, image*> pending_inputs;
};

