#pragma once

struct point
{
	int x;
	int y;
};

struct character
{
	char c;
	point pos;
	int width;
	int height;
};

struct atlas
{
	int size;
	character characters[100];
};

struct text
{
	point position;
	int width;
	int height;
};