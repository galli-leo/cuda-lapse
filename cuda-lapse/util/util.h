#pragma once
#include <string>
#include <locale>
#include <algorithm>

using namespace  std;

/**
 * \brief Converts the given string to lower case.
 * \param str 
 */
inline void lowercased(string &str)
{
	std::transform(str.begin(), str.end(), str.begin(),
		[](unsigned char c) { return tolower(c); });
}

/**
 * \brief Converts the given string to lower case and returns the lower case version.
 * \param str 
 * \return 
 */
inline string to_lowercase(string str)
{
	string copy = str;
	lowercased(copy);
	return copy;
}