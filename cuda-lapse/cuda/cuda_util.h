#pragma once
#include <unordered_map>

/**
 * \brief Print the available cuda devices including some information about them.
 * \param list_encoders Whether to list the available encoders for every cuda device.
 */
void list_cuda_devices(bool list_encoders = false);

/**
 * \brief Returns a map from device ids to available VRAM.
 * \return 
 */
std::unordered_map<int, size_t> devices_and_memory();