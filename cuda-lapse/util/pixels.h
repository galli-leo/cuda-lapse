#pragma once

/**
 * \brief Represents a single RGB pixel.
 * Used to represent a raw RGB frame in memory by using an array of rgb_pixel structs.
 */
struct rgb_pixel
{
	unsigned char red;
	unsigned char green;
	unsigned char blue;
};

/**
 * \brief Represents a single RGBA pixel.
 * Since we are on a little endian system, if we cast an rgba_pixel to an unsigned int, we will have the following mapping to bytes:
 * 
 * Byte         | 3         2         1         0
 * 
 * Repr of bits | AAAAAAAA  BBBBBBBB  GGGGGGGG  RRRRRRRR
 */
struct rgba_pixel
{
	unsigned char red;
	unsigned char green;
	unsigned char blue;
	unsigned char alpha;
};