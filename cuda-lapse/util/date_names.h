#pragma once
#include <string>
#include <unordered_map>

/**
 * \brief Mapping from weekday as an integer to a string. (0 -> Sunday, 1 -> Monday, ..., 6 -> Saturday)
 */
inline std::unordered_map<int, std::string> weekdays = {
	{0, "Sunday"},
	{1, "Monday"},
	{2, "Tuesday"},
	{3, "Wednesday"},
	{4, "Thursday"},
	{5, "Friday"},
	{6, "Saturday"},
};

/**
 * \brief Converts a weekday to its full english name. \link weekdays \endlink
 * \param day Day to convert.
 * \return Full english name of the given day.
 */
inline std::string long_weekday(int day)
{
	return weekdays[day];
}

/**
 * \brief Converts a weekday to its abbreviated english name (The first three letters).
 * \param day Day to convert.
 * \return
 */
inline std::string short_weekday(int day)
{
	return long_weekday(day).substr(0, 3);
}

/**
 * \brief Mapping from month as an integer to a string. (0 -> January, 1 -> February, ..., 11 -> December)
 */
inline std::unordered_map<int, std::string> months = {
	{0, "January"},
	{1, "February"},
	{2, "March"},
	{3, "April"},
	{4, "May"},
	{5, "June"},
	{6, "July"},
	{7, "August"},
	{8, "September"},
	{9, "October"},
	{10, "November"},
	{11, "December"},
};

/**
 * \brief Converts a month to its full english name. \link months \endlink
 * \param month Month to convert.
 * \return Full english name of the given month.
 */
inline std::string long_month(int month)
{
	return months[month];
}

/**
 * \brief Converts a month to its abbreviated english name (The first three letters).
 * \param month Month to convert.
 * \return
 */
inline std::string short_month(int month)
{
	return long_month(month).substr(0, 3);
}
